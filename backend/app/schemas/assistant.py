from typing import Literal
from pydantic import BaseModel, Field, field_validator

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)

class AssistantRequest(BaseModel):
    plant: str = Field(min_length=1, max_length=100)
    disease: str = Field(min_length=1, max_length=160)
    predicted_class: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank.")
        return value.strip()

class AssistantResponse(BaseModel):
    answer: str
    model: str
