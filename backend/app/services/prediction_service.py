from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from app.class_names import CLASS_NAMES
from app.models.vgg16_classifier import VGG16Classifier
from app.config import MAX_IMAGE_BYTES

logger = logging.getLogger("plantcare.prediction")
MIN_IMAGE_SIDE = 64
GRADCAM_MAX_SIDE = 900

def parse_class_name(class_name: str) -> tuple[str, str]:
    if "___" not in class_name:
        return "Unknown", class_name.replace("_", " ").strip()
    plant, disease = class_name.split("___", 1)
    plant = plant.replace("_", " ").replace(",", "").strip()
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy":
        disease = "Healthy"
    return plant, disease

def _image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    x = np.clip(gray.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)

class PlantDiseasePredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "VGG16"
        self.model: VGG16Classifier | None = None
        self.class_names = CLASS_NAMES.copy()
        self.image_size = 128
        self._gradcam_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.model_path}. Put vgg16_best.pth in backend/weights."
            )
        try:
            checkpoint: dict[str, Any] = torch.load(
                self.model_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format: expected a dictionary.")
        self.model_name = str(checkpoint.get("model_name", "VGG16"))
        if self.model_name != "VGG16":
            raise ValueError(f"Checkpoint architecture is {self.model_name!r}; this backend expects VGG16.")

        checkpoint_classes = checkpoint.get("class_names")
        if not isinstance(checkpoint_classes, (list, tuple)) or len(checkpoint_classes) != 38:
            raise ValueError("Checkpoint must contain the exact 38-entry class_names mapping.")
        self.class_names = list(checkpoint_classes)
        self.image_size = int(checkpoint.get("image_size", 0))
        if self.image_size <= 0:
            raise ValueError("Checkpoint image_size is missing or invalid.")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain model_state_dict.")
        self.model = VGG16Classifier(num_classes=len(self.class_names))
        # Strict loading ensures architecture/key/shape mismatches fail at startup.
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        # This is the preprocessing present in the supplied backend. The checkpoint
        # records image_size=128, but does not store its training transforms.
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        logger.info(
            "Loaded %s checkpoint (epoch=%s, val_accuracy=%s, input=%s, classes=%s) on %s",
            self.model_name, checkpoint.get("best_epoch", "unknown"),
            checkpoint.get("best_val_accuracy", "unknown"), self.image_size,
            len(self.class_names), self.device,
        )

    @staticmethod
    def decode_image(image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise ValueError("The uploaded file is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB upload limit.")
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Upload a valid JPEG, PNG or WebP image.")
                source = ImageOps.exif_transpose(source)
                source.load()
                if source.width < MIN_IMAGE_SIDE or source.height < MIN_IMAGE_SIDE:
                    raise ValueError(f"Image dimensions must be at least {MIN_IMAGE_SIDE} × {MIN_IMAGE_SIDE} pixels.")
                return source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Unable to read this image. Upload a valid JPEG, PNG or WebP image.") from exc

    def _prediction_payload(self, logits: torch.Tensor, original_size: str, top_k: int) -> dict:
        probabilities = torch.softmax(logits, dim=1)[0]
        k = min(max(int(top_k), 1), len(self.class_names))
        values, indices = torch.topk(probabilities, k=k)
        predictions = []
        for probability, index in zip(values.tolist(), indices.tolist()):
            class_name = self.class_names[index]
            plant, disease = parse_class_name(class_name)
            predictions.append({
                "class_name": class_name,
                "plant": plant,
                "disease": disease,
                "probability": float(probability),
            })
        best = predictions[0]
        confidence = best["probability"]
        return {
            "model_name": self.model_name,
            "prediction_status": "low_confidence" if confidence < 0.60 else "prediction",
            "predicted_class": best["class_name"],
            "plant": best["plant"],
            "disease": best["disease"],
            "confidence": confidence,
            "top_predictions": predictions,
            "image_size": original_size,
            "model_input_size": f"{self.image_size}x{self.image_size}",
            "disclaimer": (
                "AI screening result only. Softmax confidence is not a calibrated measure "
                "of diagnostic certainty. Confirm important crop-health decisions with a "
                "qualified agricultural expert."
            ),
        }

    @torch.inference_mode()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        image = self.decode_image(image_bytes)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        return self._prediction_payload(logits, f"{image.width}x{image.height}", top_k)

    def predict_with_gradcam(self, image_bytes: bytes, top_k: int = 3) -> dict:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        image = self.decode_image(image_bytes)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with self._gradcam_lock:
            activations: dict[str, torch.Tensor] = {}
            gradients: dict[str, torch.Tensor] = {}
            target_layer = self.model.features[28]

            def capture_activation(_module, _inputs, output):
                activations["value"] = output
                output.register_hook(lambda grad: gradients.__setitem__("value", grad))

            handle = target_layer.register_forward_hook(capture_activation)
            try:
                self.model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = self.model(tensor)
                    predicted_index = int(torch.argmax(logits, dim=1).item())
                    logits[0, predicted_index].backward()
                if "value" not in activations or "value" not in gradients:
                    raise RuntimeError("Grad-CAM failed to capture the target layer gradients.")
                feature_maps = activations["value"].detach()
                grads = gradients["value"].detach()
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
                cam = F.interpolate(
                    cam, size=(image.height, image.width),
                    mode="bilinear", align_corners=False,
                )[0, 0]
                cam = cam - cam.min()
                cam = cam / cam.max().clamp_min(1e-8)
                cam_np = cam.cpu().numpy()
            finally:
                handle.remove()
                self.model.zero_grad(set_to_none=True)

        display_image = image.copy()
        display_image.thumbnail((GRADCAM_MAX_SIDE, GRADCAM_MAX_SIDE), Image.Resampling.LANCZOS)
        cam_image = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
        cam_image = cam_image.resize(display_image.size, Image.Resampling.BILINEAR)
        heatmap = Image.fromarray(_jet_colormap(np.asarray(cam_image, dtype=np.float32) / 255.0), mode="RGB")
        overlay = Image.blend(display_image, heatmap, alpha=0.42)
        return {
            "prediction": self._prediction_payload(logits.detach(), f"{image.width}x{image.height}", top_k),
            "gradcam": {
                "heatmap_image": _image_to_data_uri(heatmap),
                "overlay_image": _image_to_data_uri(overlay),
                "explanation": (
                    "Grad-CAM highlights image regions that influenced this model's selected class. "
                    "It is an approximate visualization, not proof that the highlighted region contains the disease."
                ),
            },
        }
