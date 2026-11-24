# PlantCare AI

PlantCare AI is an AI-assisted plant leaf screening prototype with:
- VGG16 image classification (38 plant/disease categories)
- Grad-CAM heatmap and overlay visualizations
- Groq GPT-OSS 120B-powered plant-health explanations and follow-up chat

## Repository structure

- `backend/` — FastAPI + PyTorch + Grad-CAM + Groq GPT-OSS 120B
- `frontend/` — Next.js + TypeScript + Tailwind

## Run locally

### Backend (PowerShell terminal 1)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
Copy-Item .env.example .env
```

Edit `backend/.env` and set your newly rotated `GEMINI_API_KEY`.
Then start:

```powershell
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

### Frontend (PowerShell terminal 2)

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Website: http://localhost:3000

## Local CORS

The backend `.env` should include:

```env
CORS_ORIGINS=http://localhost:3000
```

## GitHub setup

Create an empty GitHub repository named `plantcare-ai` (do not add a README/license
from GitHub because this project already contains one). From the project root:

```powershell
git init
git add .
git add -f backend/weights/vgg16_best.pth
git commit -m "Build PlantCare AI full-stack application"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/plantcare-ai.git
git push -u origin main
```

The included VGG16 checkpoint is approximately 60 MB. GitHub accepts files below
100 MB but recommends Git LFS for files above 50 MB. If the final winning checkpoint
exceeds 100 MB, use a model-artifact store (for example a private Hugging Face model
repository or object storage) instead of a normal Git commit.

Never commit `backend/.env`, `frontend/.env.local`, API keys, or credentials.

## Render backend deployment

1. In Render, create a new **Web Service** connected to the GitHub repository.
2. Set **Root Directory** to `backend`.
3. Choose **Docker** runtime and use `Dockerfile`.
4. Set health check path to `/health`.
5. Add environment variables:
   - `GEMINI_API_KEY` = your newly rotated key
   - `GEMINI_MODEL` = `Groq GPT-OSS 120B-3.8-flash` (change if that model is not enabled for your key)
   - `CORS_ORIGINS` = your Vercel production URL, e.g. `https://plantcare-ai.vercel.app`
   - `MODEL_PATH` = `weights/vgg16_best.pth`
6. Deploy and test `https://YOUR-RENDER-SERVICE.onrender.com/health`.

Use a plan with enough RAM for PyTorch and VGG16. A very small instance may run
out of memory while loading the model or processing Grad-CAM.

## Vercel frontend deployment

1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Add `NEXT_PUBLIC_API_BASE_URL` = your Render service URL (no trailing slash).
4. Deploy.
5. Update Render's `CORS_ORIGINS` to the exact Vercel domain and redeploy.

## Safety and product limitations

- The classifier's softmax score is not calibrated diagnostic certainty.
- Grad-CAM is an approximate explanation of model attention, not proof of disease.
- Groq GPT-OSS 120B provides general educational guidance, not a confirmed diagnosis.
- Do not use AI output alone to decide pesticide selection or dosage. Consult
  local agricultural experts and product labels.
