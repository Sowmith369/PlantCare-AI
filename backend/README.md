# PlantCare AI — FastAPI backend

Backend for the existing Next.js frontend. It preserves these routes and payload contracts:

- `GET /health`
- `POST /api/v1/predict?top_k=3` — multipart form field `file`
- `POST /api/v1/analyze?top_k=3` — multipart form field `file`; returns `{ prediction, gradcam }`
- `POST /api/v1/assistant/chat` — JSON `{ plant, disease, predicted_class, confidence, message, history }`

The chatbot uses **Groq only**, default model `openai/gpt-oss-120b`. No Gemini integration or fallback exists.

## Windows PowerShell — local setup

Use Python 3.11. From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch. The following PyPI build supports CUDA where compatible and falls back to CPU if CUDA is unavailable:

```powershell
pip install torch==2.10.0 torchvision==0.25.0
```

Then install the API dependencies:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `GROQ_API_KEY` in `.env`. Do not send the key to anyone, commit `.env`, or put it in frontend `NEXT_PUBLIC_` variables.

Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

Frontend `.env.local` remains:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Restart the Next.js dev server after changing its environment file.

## Groq configuration

Required:
- `GROQ_API_KEY`
- `GROQ_MODEL=openai/gpt-oss-120b`

Optional:
- `GROQ_TIMEOUT_SECONDS=45`
- `GROQ_MAX_COMPLETION_TOKENS=1200`
- `GROQ_REASONING_EFFORT=medium` (`low`, `medium`, or `high` for GPT-OSS)

The API distinguishes missing credentials (503), provider rate limiting (429), upstream timeouts (504), and other provider failures (502). Provider credentials are never returned to the browser.

## Render deployment

Deploy this backend directory as a Docker web service. Set these environment variables in Render:
- `GROQ_API_KEY` — secret value
- `GROQ_MODEL` — `openai/gpt-oss-120b`
- `CORS_ORIGINS` — exact deployed Vercel origin, e.g. `https://your-app.vercel.app`
- `MODEL_PATH` — `weights/vgg16_best.pth`

Health check: `/health`.

The Dockerfile installs CPU-only PyTorch. The checkpoint is included in `weights/`; ensure your repository/provider accepts the approximately 55 MB file. Confirm the Render instance has enough RAM for PyTorch + VGG16 before production use.

## Model compatibility and preprocessing notes

The supplied checkpoint was inspected:
- `model_name`: `VGG16`
- `best_epoch`: `7`
- `best_val_accuracy`: `0.9202901986817772`
- `image_size`: `128`
- 38 class labels are stored in the checkpoint.
- The checkpoint state dictionary loaded strictly into the supplied VGG16 architecture: all keys matched.

The current inference preprocessing retains the supplied backend's `Resize(128x128)`, `ToTensor`, and ImageNet mean/std normalization. **The checkpoint does not contain training-transform metadata**, and the uploaded project did not include the training notebook/script. Therefore, this preprocessing is preserved from the previous backend, but cannot honestly be certified as identical to training preprocessing until you provide the training transform/code.

## Frontend wording

The existing frontend API request/response structure is compatible. Its UI still says “Gemini” in a few labels; change those user-facing strings to “PlantCare AI” or “Groq-powered PlantCare AI” (no API contract change). This backend does not require a frontend rewrite.

## Tests

```powershell
python -m pytest -q
```

Tests cover class-label parsing, checkpoint load, prediction output shape, Grad-CAM output, and API route contracts where practical. A live chatbot response requires your own Groq key and network access; no fake live-provider test is claimed.


## Validation performed for this handoff

- Strict checkpoint load: passed (all state-dictionary keys matched).
- Real checkpoint inference on a generated test image: passed.
- Real Grad-CAM heatmap + overlay generation: passed.
- FastAPI `/health`: passed (model_loaded=true).
- FastAPI `/api/v1/analyze`: passed (HTTP 200; `prediction` and `gradcam` returned).
- Chat endpoint without a configured key: passed expected configuration behavior (HTTP 503).
- Automated tests: 5 passed.

A live Groq completion was **not** tested because no API key was provided (and should not be shared in chat). Configure `GROQ_API_KEY` locally to test the provider call.
