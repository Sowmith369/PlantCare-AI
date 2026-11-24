from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODEL_PATH, CORS_ORIGINS, LOG_LEVEL
from app.api.prediction import router as prediction_router
from app.api.assistant import router as assistant_router
from app.services.prediction_service import PlantDiseasePredictor
from app.schemas.prediction import HealthResponse

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("plantcare")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading PlantCare AI checkpoint from %s", MODEL_PATH)
    app.state.predictor = PlantDiseasePredictor(MODEL_PATH)
    logger.info("PlantCare AI is ready on %s", app.state.predictor.device)
    try:
        yield
    finally:
        app.state.predictor = None

app = FastAPI(
    title="PlantCare AI API",
    description="VGG16 plant-leaf screening, Grad-CAM and Groq GPT-OSS 120B plant-health chat.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(prediction_router)
app.include_router(assistant_router)

@app.get("/", tags=["system"])
async def root():
    return {"service": "PlantCare AI API", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    predictor = getattr(app.state, "predictor", None)
    loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="ok" if loaded else "not_ready",
        model_loaded=loaded,
        model_name=predictor.model_name if loaded else "unavailable",
        device=str(predictor.device) if loaded else "unavailable",
    )
