import logging
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.prediction import PredictionResponse
from app.config import MAX_IMAGE_BYTES

logger = logging.getLogger("plantcare.api.prediction")
router = APIRouter(prefix="/api/v1", tags=["prediction"])

async def _read_upload(file: UploadFile) -> bytes:
    content_type = (file.content_type or "").lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image.")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB upload limit.")
    return data

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...), top_k: int = Query(default=3, ge=1, le=5)):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    try:
        data = await _read_upload(file)
        return await run_in_threadpool(predictor.predict, data, top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try another image.") from exc
    finally:
        await file.close()

@router.post("/analyze", summary="Predict and generate Grad-CAM visualizations")
async def analyze(request: Request, file: UploadFile = File(...), top_k: int = Query(default=3, ge=1, le=5)):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    try:
        data = await _read_upload(file)
        return await run_in_threadpool(predictor.predict_with_gradcam, data, top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis / Grad-CAM failed.")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try another image.") from exc
    finally:
        await file.close()
