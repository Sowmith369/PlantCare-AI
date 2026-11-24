from __future__ import annotations

import asyncio
import logging
from fastapi import HTTPException

from app.config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT_SECONDS,
    GROQ_MAX_COMPLETION_TOKENS, GROQ_REASONING_EFFORT,
)
from app.schemas.assistant import AssistantRequest

logger = logging.getLogger("plantcare.groq")

SYSTEM_INSTRUCTION = """
You are PlantCare AI, an educational plant-health information assistant.
The user's plant and condition come from an image classifier, not a confirmed
laboratory diagnosis. Describe the result as a possible condition and explain
that image classification can be wrong, especially for field photos.

Help explain symptoms, common causes, environmental conditions, general
integrated disease-management, prevention, and when to contact a local
agricultural extension officer or plant pathologist.

Safety and reliability:
- Do not invent citations, studies, product labels, or local regulations.
- Never provide exact pesticide, fungicide, or chemical dosage instructions.
- Prefer non-chemical and integrated pest-management steps where appropriate.
- For chemical control, advise following the locally approved product label
  and consulting a qualified local agricultural expert.
- Do not claim a cure is guaranteed.
- If the class is healthy, say no target disease was detected by this classifier;
  do not claim the plant is certainly healthy.
- The softmax score is not calibrated diagnostic certainty.
- Be practical, concise, and non-alarmist.
""".strip()

def build_messages(request: AssistantRequest) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for turn in request.history[-8:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({
        "role": "user",
        "content": (
            "Plant context:\n"
            f"- Plant: {request.plant}\n"
            f"- Predicted classifier label: {request.predicted_class}\n"
            f"- Readable predicted condition: {request.disease}\n"
            f"- Classifier softmax score: {request.confidence:.3f}\n\n"
            f"Current question:\n{request.message}\n\n"
            "Answer the current question using the system instructions."
        ),
    })
    return messages

async def generate_plant_answer(request: AssistantRequest) -> tuple[str, str]:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Add GROQ_API_KEY to backend/.env or the hosting environment.",
        )

    try:
        from groq import AsyncGroq, APIStatusError, APIConnectionError, APITimeoutError
    except ImportError as exc:
        logger.exception("Groq SDK is not installed.")
        raise HTTPException(status_code=503, detail="Groq SDK is missing. Install backend requirements.") from exc

    client = AsyncGroq(
        api_key=GROQ_API_KEY,
        timeout=GROQ_TIMEOUT_SECONDS,
        max_retries=0,
    )
    messages = build_messages(request)
    last_error: Exception | None = None

    try:
        for attempt in range(2):
            try:
                completion = await client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
                    reasoning_effort=GROQ_REASONING_EFFORT,
                )
                if not completion.choices:
                    raise RuntimeError("Groq returned no completion choices.")
                answer = (completion.choices[0].message.content or "").strip()
                if not answer:
                    raise RuntimeError("Groq returned an empty answer.")
                return answer, GROQ_MODEL

            except APIStatusError as exc:
                last_error = exc
                status = exc.status_code
                # Only transient provider failures are retried.
                if status in (429, 500, 502, 503, 504) and attempt == 0:
                    logger.warning("Groq transient HTTP %s; retrying once.", status)
                    await asyncio.sleep(1.0)
                    continue
                logger.error("Groq provider returned HTTP %s.", status)
                if status == 429:
                    raise HTTPException(status_code=429, detail="The AI assistant is rate-limited. Please wait a moment and try again.") from exc
                if status in (401, 403):
                    raise HTTPException(status_code=503, detail="Groq authentication or project permission failed. Check the server-side GROQ_API_KEY.") from exc
                if status in (400, 404, 422):
                    raise HTTPException(status_code=502, detail="Groq rejected the configured model request. Check GROQ_MODEL and its supported parameters.") from exc
                raise HTTPException(status_code=502, detail="Groq is temporarily unavailable. Please try again.") from exc

            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("Groq connection/timeout issue; retrying once.")
                    await asyncio.sleep(1.0)
                    continue
                logger.error("Groq connection failed after retry.")
                raise HTTPException(status_code=504, detail="Timed out while contacting Groq. Please try again.") from exc

        raise HTTPException(status_code=502, detail="Groq is temporarily unavailable. Please try again.") from last_error
    except HTTPException:
        raise
    except Exception as exc:
        # Do not log request headers, credentials, or raw provider payloads.
        logger.exception("Unexpected error while generating the Groq response.")
        raise HTTPException(status_code=502, detail="The plant-health assistant could not complete the request.") from exc
    finally:
        await client.close()
