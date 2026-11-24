from fastapi import APIRouter
from app.schemas.assistant import AssistantRequest, AssistantResponse
from app.services.groq_service import generate_plant_answer

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])

@router.post("/chat", response_model=AssistantResponse)
async def assistant_chat(request: AssistantRequest):
    answer, model_name = await generate_plant_answer(request)
    return AssistantResponse(answer=answer, model=model_name)
