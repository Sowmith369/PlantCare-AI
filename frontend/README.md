# PlantCare AI Frontend

Next.js 15 + TypeScript + Tailwind frontend.

## Local setup (Windows PowerShell)
```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```
Open http://localhost:3000. The FastAPI backend should be running at
http://127.0.0.1:8000 with `CORS_ORIGINS=http://localhost:3000`.

## Production
Set `NEXT_PUBLIC_API_BASE_URL` in Vercel to your deployed Render API URL
(for example `https://your-service.onrender.com`, without a trailing slash).

The diagnosis screen calls `POST /api/v1/analyze` for prediction + Grad-CAM,
then `POST /api/v1/assistant/chat` for Groq GPT-OSS 120B explanations and follow-up chat.
The Groq API key stays on the backend and is never exposed to the browser.
