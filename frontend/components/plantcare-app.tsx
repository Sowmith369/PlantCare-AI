"use client";

import { useCallback, useRef, useState } from "react";
import {
  Activity, ArrowRight, CheckCircle2, ChevronDown, CloudUpload,
  FileImage, Leaf, LoaderCircle, Microscope, RefreshCw, ShieldCheck,
  Sparkles, Sprout, Upload, X, AlertCircle, MessageCircle, Send, Eye
} from "lucide-react";

type ClassPrediction = {
  class_name: string;
  plant: string;
  disease: string;
  probability: number;
};

type PredictionResponse = {
  model_name: string;
  prediction_status: string;
  predicted_class: string;
  plant: string;
  disease: string;
  confidence: number;
  top_predictions: ClassPrediction[];
  image_size: string;
  model_input_size: string;
  disclaimer: string;
};

type GradCamResult = {
  heatmap_image: string;
  overlay_image: string;
  explanation: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function prettyClass(value: string) {
  return value.replaceAll("___", " · ").replaceAll("_", " ").replaceAll(",", "");
}

export default function PlantCareApp() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showDetails, setShowDetails] = useState(false);
  const [gradcam, setGradcam] = useState<GradCamResult | null>(null);
  const [visualMode, setVisualMode] = useState<"original" | "heatmap" | "overlay">("overlay");
  const [assistantMessages, setAssistantMessages] = useState<ChatMessage[]>([]);
  const [assistantInput, setAssistantInput] = useState("");
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantError, setAssistantError] = useState("");

  const clearResult = () => {
    setResult(null);
    setGradcam(null);
    setAssistantMessages([]);
    setAssistantInput("");
    setAssistantError("");
    setVisualMode("overlay");
    setError("");
    setShowDetails(false);
  };

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    clearResult();
    if (!ACCEPTED.includes(candidate.type)) {
      setError("Please choose a JPG, PNG or WebP image.");
      return;
    }
    if (candidate.size > MAX_SIZE) {
      setError("This image is larger than 10 MB. Choose a smaller image.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
    setError("");
  };

  const onDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }, [preview]);

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    clearResult();
    if (inputRef.current) inputRef.current.value = "";
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_BASE}/api/v1/analyze?top_k=3`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Analysis failed (${response.status}).`);
      }
      setResult(payload.prediction as PredictionResponse);
      setGradcam(payload.gradcam as GradCamResult);
      setVisualMode("overlay");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      if (message.includes("Failed to fetch")) {
        setError(`Could not reach the inference API at ${API_BASE}. Make sure FastAPI is running and CORS allows http://localhost:3000.`);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const askAssistant = async (messageOverride?: string) => {
    if (!result || assistantLoading) return;
    const message = (messageOverride ?? assistantInput).trim()
      || `Please explain ${result.disease} in ${result.plant}, including symptoms, causes, general management and prevention.`;
    const nextHistory = [...assistantMessages, { role: "user" as const, content: message }];
    setAssistantMessages(nextHistory);
    setAssistantInput("");
    setAssistantError("");
    setAssistantLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/assistant/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          plant: result.plant,
          disease: result.disease,
          predicted_class: result.predicted_class,
          confidence: result.confidence,
          message,
          history: assistantMessages.slice(-8),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail.map((item: { loc?: (string | number)[]; msg?: string }) =>
              `${(item.loc || []).filter((part) => part !== "body").join(".") || "Request"}: ${item.msg || "Invalid value"}`
            ).join("; ")
          : typeof payload.detail === "string" ? payload.detail : `Assistant failed (${response.status}).`;
        throw new Error(detail);
      }
      setAssistantMessages([...nextHistory, { role: "assistant", content: payload.answer }]);
    } catch (err) {
      const messageText = err instanceof Error ? err.message : "The assistant could not respond.";
      setAssistantError(messageText);
      setAssistantMessages(assistantMessages);
    } finally {
      setAssistantLoading(false);
    }
  };

  return (
    <main className="min-h-screen overflow-hidden bg-cream text-forest-950">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="relative z-10 mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-6 sm:px-8 lg:px-12">
        <a href="#" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-forest-900 text-white shadow-lg shadow-forest-900/15">
            <Sprout size={23} strokeWidth={1.8} />
          </div>
          <div>
            <div className="font-display text-xl font-bold tracking-tight">PlantCare <span className="text-leaf-600">AI</span></div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-forest-800/55">Plant health, made clearer</div>
          </div>
        </a>
        <div className="hidden items-center gap-7 text-sm font-medium text-forest-900/65 md:flex">
          <a href="#how-it-works" className="transition hover:text-leaf-600">How it works</a>
          <a href="#diagnose" className="transition hover:text-leaf-600">Disease checker</a>
          <span className="rounded-full border border-leaf-200 bg-white/70 px-3 py-1.5 text-xs text-forest-800">Research prototype</span>
        </div>
        <a href="#diagnose" className="rounded-full bg-forest-900 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-forest-900/10 transition hover:bg-forest-800 sm:px-5 sm:text-sm">
          Try the checker <ArrowRight className="ml-1 inline" size={15} />
        </a>
      </header>

      <section className="relative z-10 mx-auto grid max-w-7xl items-center gap-12 px-5 pb-16 pt-10 sm:px-8 md:pt-16 lg:grid-cols-[1.03fr_.97fr] lg:px-12 lg:pb-24">
        <div className="max-w-2xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-leaf-200 bg-white/75 px-3.5 py-2 text-xs font-semibold text-forest-800 shadow-sm">
            <span className="h-2 w-2 rounded-full bg-leaf-500 ring-4 ring-leaf-100" />
            AI-assisted plant health screening
          </div>
          <h1 className="font-display text-[3.25rem] font-semibold leading-[1.04] tracking-[-0.055em] text-forest-950 sm:text-6xl lg:text-[4.45rem]">
            A healthier harvest starts with <span className="text-leaf-600">one leaf.</span>
          </h1>
          <p className="mt-6 max-w-xl text-base leading-8 text-forest-900/65 sm:text-lg">
            Upload a photo of a plant leaf to screen for common diseases. Get a clear prediction and a starting point for understanding what your plant may need.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <a href="#diagnose" className="inline-flex items-center gap-2 rounded-full bg-forest-900 px-6 py-3.5 text-sm font-semibold text-white shadow-xl shadow-forest-900/15 transition hover:-translate-y-0.5 hover:bg-forest-800">
              Check a leaf <ArrowRight size={17} />
            </a>
            <a href="#how-it-works" className="inline-flex items-center gap-2 rounded-full border border-forest-900/10 bg-white/65 px-6 py-3.5 text-sm font-semibold text-forest-900 transition hover:bg-white">
              See how it works
            </a>
          </div>
          <div className="mt-10 flex flex-wrap gap-x-6 gap-y-3 text-xs font-medium text-forest-900/55">
            <span className="inline-flex items-center gap-2"><ShieldCheck size={16} className="text-leaf-600" /> Image-based screening</span>
            <span className="inline-flex items-center gap-2"><Microscope size={16} className="text-leaf-600" /> 38 trained categories</span>
            <span className="inline-flex items-center gap-2"><Activity size={16} className="text-leaf-600" /> AI-assisted insights</span>
          </div>
        </div>

        <div className="relative mx-auto w-full max-w-[570px]">
          <div className="absolute -left-7 top-12 h-32 w-32 rounded-full bg-leaf-200/70 blur-3xl" />
          <div className="absolute -right-5 bottom-8 h-40 w-40 rounded-full bg-amber-100/70 blur-3xl" />
          <div className="hero-image-wrap relative overflow-hidden rounded-[2rem] border border-white/80 bg-[#e7efdf] p-3 shadow-soft">
            <div className="relative overflow-hidden rounded-[1.5rem]">
              <div className="relative flex h-[390px] w-full items-center justify-center overflow-hidden bg-[radial-gradient(ellipse_at_65%_35%,#b8d0a6_0%,#759b68_34%,#315d43_72%,#173b2c_100%)] sm:h-[470px]">
                <div className="absolute -left-16 top-12 h-64 w-64 rounded-full border border-white/15" />
                <div className="absolute -right-20 bottom-0 h-80 w-80 rounded-full border border-white/15" />
                <div className="absolute left-1/2 top-1/2 h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/10 blur-2xl" />
                <Leaf className="relative z-10 h-56 w-56 -rotate-[28deg] text-[#e4f0d7]/90 drop-shadow-2xl sm:h-72 sm:w-72" strokeWidth={0.7} />
                <div className="absolute bottom-16 left-10 h-32 w-32 rotate-[-35deg] rounded-[100%_0_100%_0] border border-white/20 bg-white/5" />
                <div className="absolute right-12 top-24 h-24 w-24 rotate-[25deg] rounded-[100%_0_100%_0] border border-white/20 bg-white/5" />
              </div>
              <div className="absolute inset-0 bg-gradient-to-t from-forest-950/45 via-transparent to-forest-950/5" />
              <div className="absolute left-5 top-5 rounded-full border border-white/30 bg-white/85 px-3 py-2 text-[11px] font-semibold text-forest-900 backdrop-blur">
                <span className="mr-2 inline-block h-2 w-2 rounded-full bg-leaf-500" /> Nature, meet intelligence
              </div>
              <div className="absolute bottom-5 left-5 right-5 rounded-2xl border border-white/40 bg-white/90 p-4 shadow-xl backdrop-blur-md">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-forest-800/50">A clearer first step</div>
                    <div className="mt-1 font-display text-lg font-semibold text-forest-950">Understand what you see.</div>
                  </div>
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-leaf-100 text-leaf-600"><Leaf size={22}/></div>
                </div>
              </div>
            </div>
          </div>
          <div className="absolute -bottom-5 -left-5 hidden items-center gap-3 rounded-2xl border border-white bg-white px-4 py-3 shadow-soft sm:flex">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-leaf-50 text-leaf-600"><Sparkles size={19}/></div>
            <div><div className="text-xs font-bold text-forest-950">AI-powered screening</div><div className="text-[10px] text-forest-900/50">A helpful place to begin</div></div>
          </div>
        </div>
      </section>

      <section id="diagnose" className="relative z-10 scroll-mt-8 bg-white py-20 sm:py-24">
        <div className="mx-auto max-w-7xl px-5 sm:px-8 lg:px-12">
          <div className="mx-auto max-w-2xl text-center">
            <div className="section-kicker"><span className="h-1.5 w-1.5 rounded-full bg-leaf-500"/> LEAF DISEASE CHECKER</div>
            <h2 className="mt-4 font-display text-3xl font-semibold tracking-tight text-forest-950 sm:text-4xl">Let’s take a closer look.</h2>
            <p className="mt-4 text-sm leading-7 text-forest-900/60 sm:text-base">Choose a clear photo of a single leaf. Our image classifier will compare it with the disease categories it was trained to recognize.</p>
          </div>

          <div className="mx-auto mt-10 grid max-w-4xl grid-cols-1 gap-6 sm:mt-12 sm:gap-8">
            <div className="rounded-[1.75rem] border border-leaf-100 bg-cream p-5 shadow-sm sm:p-7">
              <div className="mb-5 flex items-center justify-between">
                <div><h3 className="font-display text-lg font-semibold">Upload leaf image</h3><p className="mt-1 text-xs text-forest-900/50">JPG, PNG or WebP · up to 10 MB</p></div>
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-leaf-100 text-leaf-600"><Upload size={19}/></div>
              </div>

              <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(e) => chooseFile(e.target.files?.[0])} />
              {!preview ? (
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={onDrop}
                  onClick={() => inputRef.current?.click()}
                  className={`group flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 text-center transition ${dragging ? "border-leaf-600 bg-leaf-100" : "border-leaf-200 bg-white hover:border-leaf-500 hover:bg-leaf-50/60"}`}
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-leaf-50 text-leaf-600 transition group-hover:scale-105"><CloudUpload size={30} strokeWidth={1.6}/></div>
                  <p className="mt-5 text-sm font-semibold text-forest-950">Drop your leaf image here</p>
                  <p className="mt-2 text-xs text-forest-900/45">or browse files from your device</p>
                  <span className="mt-6 rounded-full bg-forest-900 px-5 py-2.5 text-xs font-semibold text-white transition group-hover:bg-forest-800">Choose image</span>
                </div>
              ) : (
                <div className="relative overflow-hidden rounded-2xl border border-leaf-100 bg-white">
                  <div className="flex min-h-[240px] items-center justify-center bg-[#edf2e9] p-3 sm:min-h-[300px]">
                    <img src={preview} alt="Selected leaf preview" className="max-h-[390px] w-full rounded-xl object-contain" />
                  </div>
                  <div className="flex items-center justify-between gap-3 border-t border-leaf-100 px-4 py-3">
                    <div className="flex min-w-0 items-center gap-2"><FileImage size={17} className="shrink-0 text-leaf-600"/><span className="truncate text-xs font-medium text-forest-900">{file?.name}</span></div>
                    <button onClick={reset} className="rounded-lg p-2 text-forest-900/50 transition hover:bg-red-50 hover:text-red-600" aria-label="Remove image"><X size={17}/></button>
                  </div>
                </div>
              )}

              {error && <div className="mt-4 flex items-start gap-2 rounded-xl border border-red-100 bg-red-50 px-4 py-3 text-xs leading-5 text-red-700"><AlertCircle size={16} className="mt-0.5 shrink-0"/>{error}</div>}

              <button disabled={!file || loading} onClick={analyze} className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-forest-900 px-5 py-4 text-sm font-semibold text-white shadow-lg shadow-forest-900/10 transition hover:bg-forest-800 disabled:cursor-not-allowed disabled:opacity-40">
                {loading ? <><LoaderCircle size={18} className="animate-spin"/> Analyzing your image…</> : <><Sparkles size={17}/> Analyze leaf image <ArrowRight size={16}/></>}
              </button>
              <p className="mt-3 text-center text-[10px] leading-5 text-forest-900/45">Your image is sent to the PlantCare AI inference service for analysis.</p>
            </div>

            <div className="rounded-[1.75rem] border border-forest-900/5 bg-forest-950 p-5 text-white shadow-soft sm:p-7">
              {!result && !loading ? (
                <div className="flex h-full min-h-[400px] flex-col">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-leaf-200"><Activity size={16}/> Your result</div>
                  <div className="my-auto py-12 text-center">
                    <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-[1.7rem] border border-white/10 bg-white/5 text-leaf-200"><Microscope size={34} strokeWidth={1.4}/></div>
                    <h3 className="mt-6 font-display text-xl font-semibold">Your diagnosis will appear here</h3>
                    <p className="mx-auto mt-3 max-w-xs text-xs leading-6 text-white/50">Upload a leaf photo and run the analysis to see the predicted plant disease and model confidence.</p>
                  </div>
                  <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-[11px] text-white/55"><ShieldCheck size={16} className="text-leaf-200"/> Screening support, not a substitute for expert diagnosis.</div>
                </div>
              ) : loading ? (
                <div className="flex min-h-[400px] flex-col items-center justify-center text-center">
                  <div className="relative flex h-20 w-20 items-center justify-center rounded-full border border-leaf-200/20 bg-white/5"><LoaderCircle size={35} className="animate-spin text-leaf-200"/></div>
                  <h3 className="mt-6 font-display text-xl font-semibold">Reading the leaf…</h3>
                  <p className="mt-2 text-xs text-white/50">The image classifier is comparing visual patterns.</p>
                  <div className="mt-8 h-1.5 w-48 overflow-hidden rounded-full bg-white/10"><div className="h-full w-2/3 animate-pulse rounded-full bg-leaf-500"/></div>
                </div>
              ) : result ? (
                <div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-leaf-200"><CheckCircle2 size={16}/> Screening result</div>
                    <button onClick={reset} className="inline-flex items-center gap-1 text-[11px] text-white/50 hover:text-white"><RefreshCw size={13}/> New image</button>
                  </div>
                  <div className="mt-8">
                    <div className="text-xs text-white/45">Predicted condition</div>
                    <h3 className="mt-2 font-display text-3xl font-semibold leading-tight">{result.disease}</h3>
                    <p className="mt-2 text-sm text-leaf-200">{result.plant}</p>
                  </div>
                  <div className="mt-7 rounded-2xl border border-white/10 bg-white/[0.06] p-4">
                    <div className="flex items-end justify-between"><div><div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/45">Model score</div><div className="mt-1 text-3xl font-semibold tabular-nums">{percent(result.confidence)}</div></div><span className="rounded-full bg-leaf-500/15 px-2.5 py-1 text-[10px] font-semibold text-leaf-200">{result.prediction_status === "low_confidence" ? "Low score" : "Top match"}</span></div>
                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-leaf-400 transition-all" style={{ width: percent(result.confidence) }}/></div>
                    <p className="mt-3 text-[10px] leading-5 text-white/45">Softmax score; not a calibrated measure of diagnostic certainty.</p>
                  </div>
                  <div className="mt-7">
                    <button onClick={() => setShowDetails(!showDetails)} className="flex w-full items-center justify-between text-left text-xs font-semibold text-white/80"><span>Other likely matches</span><ChevronDown size={16} className={`transition ${showDetails ? "rotate-180" : ""}`}/></button>
                    <div className="mt-3 space-y-3">
                      {(showDetails ? result.top_predictions : result.top_predictions.slice(0, 3)).map((item, index) => (
                        <div key={item.class_name} className="flex items-center gap-3">
                          <span className="w-5 text-[10px] text-white/35">0{index + 1}</span>
                          <div className="min-w-0 flex-1"><div className="truncate text-xs font-medium text-white/85">{prettyClass(item.class_name)}</div><div className="mt-1 h-1 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-leaf-400/80" style={{ width: percent(item.probability) }}/></div></div>
                          <span className="text-xs tabular-nums text-white/60">{percent(item.probability)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="mt-7 rounded-xl border border-white/10 bg-white/5 p-3 text-[10px] leading-5 text-white/50">{result.disclaimer}</div>
                  <div className="mt-4 flex items-center gap-2 text-[10px] text-white/35"><CheckCircle2 size={13}/> Model: {result.model_name} · Original: {result.image_size} · Model input: {result.model_input_size}</div>

                  {gradcam && (
                    <div className="mt-7 border-t border-white/10 pt-6">
                      <div className="flex items-center gap-2 text-xs font-semibold text-white/90"><Eye size={15} className="text-leaf-200"/> Model attention visualization</div>
                      <p className="mt-2 text-[10px] leading-5 text-white/45">{gradcam.explanation}</p>
                      <div className="mt-4 grid grid-cols-3 gap-2">
                        {([
                          ["original", "Original", preview],
                          ["heatmap", "Heatmap", gradcam.heatmap_image],
                          ["overlay", "Overlay", gradcam.overlay_image],
                        ] as const).map(([mode, label, src]) => (
                          <button key={mode} onClick={() => setVisualMode(mode)} className={`overflow-hidden rounded-xl border text-left transition ${visualMode === mode ? "border-leaf-300 ring-2 ring-leaf-300/20" : "border-white/10 opacity-75 hover:opacity-100"}`}>
                            {src ? <img src={src} alt={`${label} Grad-CAM view`} className="h-20 w-full object-cover" /> : <div className="h-20 w-full bg-white/10"/>}
                            <span className="block px-2 py-2 text-center text-[10px] font-semibold text-white/80">{label}</span>
                          </button>
                        ))}
                      </div>
                      <div className="mt-3 overflow-hidden rounded-xl border border-white/10 bg-black/10">
                        {visualMode === "original" && preview && <img src={preview} alt="Original uploaded leaf" className="max-h-72 w-full object-contain"/>}
                        {visualMode === "heatmap" && <img src={gradcam.heatmap_image} alt="Grad-CAM heatmap" className="max-h-72 w-full object-contain"/>}
                        {visualMode === "overlay" && <img src={gradcam.overlay_image} alt="Grad-CAM overlay" className="max-h-72 w-full object-contain"/>}
                      </div>
                    </div>
                  )}

                  <div className="mt-7 border-t border-white/10 pt-6">
                    <div className="flex items-center gap-2 text-xs font-semibold text-white/90"><Sparkles size={16} className="text-leaf-200"/> PlantCare AI assistant</div>
                    <p className="mt-2 text-[10px] leading-5 text-white/45">Ask PlantCare AI to explain this classifier result. The prediction is a screening result, not a confirmed diagnosis.</p>
                    {assistantMessages.length === 0 ? (
                      <button disabled={assistantLoading} onClick={() => askAssistant()} className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-leaf-100 px-4 py-3 text-xs font-bold text-forest-950 transition hover:bg-leaf-200 disabled:opacity-50">
                        {assistantLoading ? <LoaderCircle size={15} className="animate-spin"/> : <Sparkles size={15}/>}
                        Explain this condition with PlantCare AI
                      </button>
                    ) : (
                      <div className="mt-4 max-h-[560px] space-y-3 overflow-y-auto pr-1 sm:max-h-[680px]">
                        {assistantMessages.map((message, index) => (
                          <div key={`${message.role}-${index}`} className={`rounded-xl px-3 py-3 text-xs leading-6 ${message.role === "user" ? "ml-7 bg-white/10 text-white/90" : "mr-3 border border-leaf-200/15 bg-leaf-100/10 text-white/85"}`}>
                            <div className="mb-1 text-[9px] font-bold uppercase tracking-wider text-leaf-200">{message.role === "user" ? "You" : "PlantCare AI · Groq"}</div>
                            <p className="whitespace-pre-wrap">{message.content}</p>
                          </div>
                        ))}
                        {assistantLoading && <div className="flex items-center gap-2 text-xs text-leaf-100/70"><LoaderCircle size={14} className="animate-spin"/> PlantCare AI is preparing an answer…</div>}
                      </div>
                    )}
                    {assistantError && <p className="mt-3 rounded-lg bg-red-400/10 px-3 py-2 text-[10px] leading-5 text-red-200">{assistantError}</p>}
                    <form onSubmit={(event) => { event.preventDefault(); void askAssistant(); }} className="mt-4 flex w-full items-stretch gap-2">
                      <input value={assistantInput} onChange={(event) => setAssistantInput(event.target.value)} placeholder="Ask a follow-up question…" maxLength={1200} className="min-w-0 flex-1 rounded-xl border border-white/15 bg-white/10 px-3 py-3 text-xs text-white outline-none placeholder:text-white/35 focus:border-leaf-200"/>
                      <button type="submit" disabled={assistantLoading || !assistantInput.trim()} aria-label="Send question" className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-leaf-100 text-forest-950 transition hover:bg-leaf-200 disabled:opacity-40"><Send size={16}/></button>
                    </form>
                    <p className="mt-2 text-[9px] leading-4 text-white/35">PlantCare AI responses are AI-generated general guidance. Verify treatment decisions with a qualified local agricultural expert.</p>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </section>

      <section id="how-it-works" className="relative z-10 mx-auto max-w-7xl px-5 py-20 sm:px-8 lg:px-12 lg:py-24">
        <div className="mx-auto max-w-2xl text-center">
          <div className="section-kicker"><span className="h-1.5 w-1.5 rounded-full bg-leaf-500"/> SIMPLE BY DESIGN</div>
          <h2 className="mt-4 font-display text-3xl font-semibold tracking-tight sm:text-4xl">From leaf photo to a clearer picture.</h2>
          <p className="mt-4 text-sm leading-7 text-forest-900/60">A simple first step toward understanding what may be affecting your plant.</p>
        </div>
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {[
            { n: "01", icon: Upload, title: "Share a photo", body: "Choose a clear image of the affected leaf, preferably in natural light." },
            { n: "02", icon: Microscope, title: "Screen the leaf", body: "The trained image classifier compares the photo against its known categories." },
            { n: "03", icon: Sparkles, title: "Explore the result", body: "Review the prediction, explore Grad-CAM visualizations and ask the PlantCare AI assistant." },
          ].map((item) => (
            <div key={item.n} className="rounded-3xl border border-leaf-100 bg-white/75 p-7 shadow-sm">
              <div className="flex items-center justify-between"><div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-leaf-50 text-leaf-600"><item.icon size={22}/></div><span className="font-display text-3xl font-semibold text-leaf-200">{item.n}</span></div>
              <h3 className="mt-7 font-display text-xl font-semibold">{item.title}</h3>
              <p className="mt-3 text-sm leading-7 text-forest-900/55">{item.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="relative z-10 px-5 pb-16 sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-7xl flex-col justify-between gap-6 rounded-[2rem] bg-forest-900 px-7 py-9 text-white sm:px-10 md:flex-row md:items-center">
          <div><div className="text-xs font-semibold uppercase tracking-[0.18em] text-leaf-200">A thoughtful first step</div><h2 className="mt-3 font-display text-2xl font-semibold sm:text-3xl">Let’s help your plants thrive.</h2><p className="mt-2 max-w-xl text-sm leading-6 text-white/55">Start with a leaf photo. Keep an eye on your plants, and seek local expert advice for important crop decisions.</p></div>
          <a href="#diagnose" className="inline-flex shrink-0 items-center justify-center gap-2 rounded-full bg-white px-6 py-3.5 text-sm font-semibold text-forest-900 transition hover:bg-leaf-50">Check a leaf <ArrowRight size={16}/></a>
        </div>
      </section>

      <footer className="relative z-10 border-t border-forest-900/5 px-5 py-7 sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 text-xs text-forest-900/45 sm:flex-row sm:items-center sm:justify-between">
          <div className="font-display text-sm font-bold text-forest-900">PlantCare <span className="text-leaf-600">AI</span></div>
          <p>AI-assisted screening for educational purposes. Not a substitute for professional agricultural diagnosis.</p>
          <p>© {new Date().getFullYear()} PlantCare AI</p>
        </div>
      </footer>
    </main>
  );
}
