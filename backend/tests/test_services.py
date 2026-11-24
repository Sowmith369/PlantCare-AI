import io
from PIL import Image
import pytest

from app.services.prediction_service import parse_class_name, PlantDiseasePredictor
from app.config import MODEL_PATH
from app.schemas.assistant import AssistantRequest
from app.services.groq_service import build_messages

def test_parse_tomato_early_blight():
    assert parse_class_name("Tomato___Early_blight") == ("Tomato", "Early blight")

def test_parse_healthy_class():
    assert parse_class_name("Apple___healthy") == ("Apple", "Healthy")

def test_parse_corn_class():
    assert parse_class_name("Corn_(maize)___Common_rust_") == ("Corn (maize)", "Common rust")

def test_groq_messages_include_context_and_history():
    req = AssistantRequest(
        plant="Tomato", disease="Early blight", predicted_class="Tomato___Early_blight",
        confidence=0.91, message="How can I prevent it?",
        history=[{"role": "user", "content": "What is it?"}, {"role": "assistant", "content": "A possible fungal disease."}],
    )
    messages = build_messages(req)
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Tomato___Early_blight" in messages[-1]["content"]
    assert messages[1]["content"] == "What is it?"

@pytest.mark.integration
def test_real_checkpoint_predict_and_gradcam():
    if not MODEL_PATH.exists():
        pytest.skip("Checkpoint is not present.")
    predictor = PlantDiseasePredictor(MODEL_PATH)
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (40, 150, 70)).save(buffer, format="PNG")
    data = buffer.getvalue()

    prediction = predictor.predict(data, top_k=3)
    assert len(prediction["top_predictions"]) == 3
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["model_input_size"] == "128x128"

    result = predictor.predict_with_gradcam(data, top_k=3)
    assert set(result) == {"prediction", "gradcam"}
    assert result["gradcam"]["heatmap_image"].startswith("data:image/png;base64,")
    assert result["gradcam"]["overlay_image"].startswith("data:image/png;base64,")


def test_chat_history_accepts_long_assistant_reply():
    # GPT-OSS answers can exceed 1,200 characters; they must remain valid
    # when sent back as conversation history on the next user turn.
    long_reply = "Plant-care explanation. " * 250
    request = AssistantRequest(
        plant="Apple", disease="Cedar apple rust",
        predicted_class="Apple___Cedar_apple_rust",
        confidence=0.625, message="What should I do next?",
        history=[
            {"role": "user", "content": "Explain this disease."},
            {"role": "assistant", "content": long_reply},
        ],
    )
    assert len(request.history[1].content) > 1200
