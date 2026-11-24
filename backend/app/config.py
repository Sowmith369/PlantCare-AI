from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip().rstrip("/") for item in os.getenv(name, default).split(",") if item.strip()]

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "weights" / "vgg16_best.pth")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BASE_DIR / MODEL_PATH).resolve()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "45"))
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "1200"))
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "medium").strip()

MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024
CORS_ORIGINS = _csv_env("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
