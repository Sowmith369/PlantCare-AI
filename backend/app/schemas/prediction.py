from pydantic import BaseModel, Field

class ClassPrediction(BaseModel):
    class_name: str
    plant: str
    disease: str
    probability: float = Field(ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    model_name: str
    prediction_status: str
    predicted_class: str
    plant: str
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[ClassPrediction]
    image_size: str
    model_input_size: str
    disclaimer: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
