"""
CareVoice Trimodal Clinical Intake Server — Gemma 4 4B-IT
==========================================================
Modalities: text | image (base64 PNG/JPEG) | audio (base64 WAV 16kHz mono)

Architecture: Gemma4ForConditionalGeneration  (vision_config + audio_config + text_config)
VRAM budget:  ~15.9 GB on L40S 48GB

Endpoints:
  GET  /health
  POST /generate        — text-only conversation
  POST /generate_image  — text + clinical image (wound/skin/X-ray/pill)
  POST /generate_audio  — text + audio clip (16kHz mono WAV)
  POST /triage_image    — alias for /generate_image with explicit triage framing
"""

import gc
import sys
import types
import json
import re
import os
import io
import base64
import logging
import threading
from typing import List, Dict, Optional, Any

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("carevoice")

# ─── Video-utils stub (not needed at runtime) ────────────────────────────────
_video_stub = types.ModuleType("video_utils")
_video_stub.load_video = lambda *a, **kw: []
sys.modules.setdefault("video_utils", _video_stub)

# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH        = os.environ.get("MODEL_PATH", "/workspace/gemma4")
MAX_NEW_TOKENS    = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE       = float(os.environ.get("TEMPERATURE", "0.3"))
PORT              = int(os.environ.get("PORT", "8000"))
AUDIO_SAMPLE_RATE = 16000   # Gemma 4 audio encoder: 16 kHz mono

# ─── System prompts ──────────────────────────────────────────────────────────

TEXT_SYSTEM_PROMPT = """You are CareVoice, an AI clinical intake assistant. Your role:
1. Gather patient symptoms empathetically (text, visual, or audio input)
2. Identify chief complaint, severity, duration
3. Detect RED FLAG symptoms requiring URGENT escalation (set urgent=true):
   - Chest pain/pressure, difficulty breathing, shortness of breath
   - Sudden severe headache ("worst headache of life"), thunderclap onset
   - Sudden one-sided weakness/numbness, facial droop, slurred speech (stroke)
   - Suicidal ideation or active self-harm plan
   - Throat swelling + hives (anaphylaxis), uncontrolled bleeding, loss of consciousness
   - Severe abdominal pain in pregnancy, diabetic crisis (glucose > 400)

Always respond in valid JSON:
{
  "response": "<empathetic message to patient>",
  "extracted_info": {
    "chief_complaint": "<main complaint or null>",
    "symptoms": ["<list>"],
    "duration": "<duration or null>",
    "severity": "mild|moderate|severe|null",
    "medications": ["<list>"],
    "allergies": ["<list>"],
    "urgent": true|false,
    "escalation_reason": "<reason or null>",
    "triage_level": "green|yellow|red|null"
  },
  "intake_complete": true|false
}"""

IMAGE_SYSTEM_PROMPT = """You are CareVoice, an AI clinical triage assistant analyzing a patient-submitted medical image.

Analyze the image carefully. For skin/wound images: assess appearance, severity, urgency. For X-rays: identify visible pathology. For medication/pill images: identify the medication.

triage_level definitions:
  green  = routine care, no urgency
  yellow = prompt evaluation within hours
  red    = immediate emergency care required

Respond ONLY in valid JSON:
{
  "response": "<empathetic clinical explanation for the patient>",
  "visual_findings": {
    "image_type": "wound|skin_lesion|xray|medication|eye|other",
    "description": "<clinical description of visible findings>",
    "severity_indicators": ["<list of concerning features>"],
    "differential": ["<top 2-3 clinical possibilities>"]
  },
  "extracted_info": {
    "chief_complaint": "<visual complaint>",
    "symptoms": ["<inferred symptoms>"],
    "duration": null,
    "severity": "mild|moderate|severe",
    "medications": [],
    "allergies": [],
    "urgent": true|false,
    "escalation_reason": "<reason or null>",
    "triage_level": "green|yellow|red"
  },
  "intake_complete": false,
  "follow_up_questions": ["<1-2 questions to complete intake>"]
}"""

AUDIO_SYSTEM_PROMPT = """You are CareVoice, an AI clinical intake assistant listening to a patient's audio.

If the audio is speech: transcribe it and extract symptoms.
If the audio is a respiratory sound (cough, wheeze, breathing): describe it clinically.

Respond in valid JSON:
{
  "response": "<empathetic response acknowledging what was heard>",
  "audio_analysis": {
    "audio_type": "speech|cough|breathing|wheeze|other",
    "transcription": "<verbatim if speech, else null>",
    "clinical_observations": "<clinical description of audio findings>",
    "respiratory_findings": {
      "cough_present": true|false,
      "wheeze_present": true|false,
      "stridor_present": true|false,
      "abnormal_breathing": true|false
    }
  },
  "extracted_info": {
    "chief_complaint": "<complaint from audio>",
    "symptoms": ["<extracted symptoms>"],
    "duration": "<if mentioned>",
    "severity": "mild|moderate|severe|null",
    "medications": [],
    "allergies": [],
    "urgent": true|false,
    "escalation_reason": "<if urgent>",
    "triage_level": "green|yellow|red|null"
  },
  "intake_complete": false
}"""

# ─── Model loading ────────────────────────────────────────────────────────────
log.info(f"Loading Gemma4ForConditionalGeneration from {MODEL_PATH} ...")

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"VRAM: {used:.1f} GB / {total:.1f} GB | GPU: {torch.cuda.get_device_name(0)}")

log.info("Model + processor loaded. Modalities: text, image, audio.")

# Serialise all GPU generate() calls — prevents concurrent CUDA index assertion errors
_GPU_LOCK = threading.Lock()

# ─── Utility helpers ─────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_RAW   = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(text: str) -> Optional[Dict]:
    for pat in (_JSON_BLOCK, _JSON_RAW):
        m = pat.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


def safe_output(raw: str, image_mode: bool = False) -> Dict:
    parsed = extract_json(raw)
    if parsed:
        return parsed
    base: Dict[str, Any] = {
        "response": raw.strip(),
        "extracted_info": {
            "chief_complaint": None, "symptoms": [],
            "duration": None, "severity": None,
            "medications": [], "allergies": [],
            "urgent": False, "escalation_reason": None, "triage_level": None,
        },
        "intake_complete": False,
    }
    if image_mode:
        base["visual_findings"] = {
            "image_type": "other", "description": raw.strip(),
            "severity_indicators": [], "differential": [],
        }
    return base


def decode_image(b64_str: str) -> Image.Image:
    """Decode base64 image — strips data-URL prefix if present."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def decode_audio(b64_str: str) -> np.ndarray:
    """Decode base64 WAV → float32 numpy array resampled to AUDIO_SAMPLE_RATE."""
    import wave
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data   = base64.b64decode(b64_str)
    buf    = io.BytesIO(data)
    try:
        with wave.open(buf, "rb") as wf:
            sr       = wf.getframerate()
            nframes  = wf.getnframes()
            raw_pcm  = wf.readframes(nframes)
            nchan    = wf.getnchannels()
            sampw    = wf.getsampwidth()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampw, np.int16)
        arr   = np.frombuffer(raw_pcm, dtype=dtype).astype(np.float32)
        # Stereo → mono
        if nchan > 1:
            arr = arr.reshape(-1, nchan).mean(axis=1)
        # Normalise to [-1, 1]
        max_val = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
        arr /= max_val
        # Resample if needed
        if sr != AUDIO_SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g   = gcd(AUDIO_SAMPLE_RATE, sr)
                arr = resample_poly(arr, AUDIO_SAMPLE_RATE // g, sr // g)
            except ImportError:
                ratio   = AUDIO_SAMPLE_RATE / sr
                new_len = int(len(arr) * ratio)
                arr     = np.interp(
                    np.linspace(0, len(arr) - 1, new_len),
                    np.arange(len(arr)), arr,
                ).astype(np.float32)
        return arr.astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Audio decode failed: {exc}")


# ─── Inference ───────────────────────────────────────────────────────────────

def run_text_inference(conversation: List[Dict[str, str]]) -> Dict:
    messages = [{"role": "system", "content": TEXT_SYSTEM_PROMPT}] + conversation
    prompt   = processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs    = processor(text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with _GPU_LOCK:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    raw = processor.decode(out[0, input_len:], skip_special_tokens=True)
    return safe_output(raw)


def run_image_inference(image: Image.Image, text_context: str = "") -> Dict:
    user_text = text_context.strip() or "Please analyze this clinical image and provide a triage assessment."
    messages  = [
        {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
        {"role": "user",   "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": user_text},
        ]},
    ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs    = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with _GPU_LOCK:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    raw = processor.decode(out[0, input_len:], skip_special_tokens=True)
    return safe_output(raw, image_mode=True)


def run_audio_inference(audio_arr: np.ndarray, text_context: str = "") -> Dict:
    user_text = text_context.strip() or "Please analyze this clinical audio recording."
    messages  = [
        {"role": "system", "content": AUDIO_SYSTEM_PROMPT},
        {"role": "user",   "content": [
            {"type": "audio"},
            {"type": "text", "text": user_text},
        ]},
    ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs    = processor(
        text=prompt,
        audios=[(audio_arr, AUDIO_SAMPLE_RATE)],
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with _GPU_LOCK:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    raw = processor.decode(out[0, input_len:], skip_special_tokens=True)
    return safe_output(raw)


# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="CareVoice Trimodal Server", version="2.0.0")


class TextRequest(BaseModel):
    conversation:   List[Dict[str, str]]
    max_new_tokens: Optional[int]   = None
    temperature:    Optional[float] = None


class ImageRequest(BaseModel):
    image_b64:      str                    # base64 PNG or JPEG
    text_context:   Optional[str]  = ""
    max_new_tokens: Optional[int]  = None


class AudioRequest(BaseModel):
    audio_b64:      str                    # base64 16kHz mono WAV
    text_context:   Optional[str]  = ""
    max_new_tokens: Optional[int]  = None


@app.get("/health")
def health():
    info: Dict[str, Any] = {
        "status":     "ok",
        "model":      "gemma4-4b-it",
        "modalities": ["text", "image", "audio"],
        "version":    "2.0.0",
    }
    if torch.cuda.is_available():
        info.update({
            "gpu":           torch.cuda.get_device_name(0),
            "vram_used_gb":  round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        })
    return info


@app.post("/generate")
def generate(req: TextRequest):
    global MAX_NEW_TOKENS, TEMPERATURE
    if req.max_new_tokens is not None: MAX_NEW_TOKENS = req.max_new_tokens
    if req.temperature    is not None: TEMPERATURE    = req.temperature
    try:
        return JSONResponse(content=run_text_inference(req.conversation))
    except Exception as exc:
        log.exception("Text inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate_image")
def generate_image(req: ImageRequest):
    global MAX_NEW_TOKENS
    if req.max_new_tokens is not None: MAX_NEW_TOKENS = req.max_new_tokens
    try:
        img = decode_image(req.image_b64)
        return JSONResponse(content=run_image_inference(img, req.text_context or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Image inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/triage_image")
def triage_image(req: ImageRequest):
    """Explicit triage alias — same as /generate_image."""
    return generate_image(req)


@app.post("/generate_audio")
def generate_audio(req: AudioRequest):
    global MAX_NEW_TOKENS
    if req.max_new_tokens is not None: MAX_NEW_TOKENS = req.max_new_tokens
    try:
        arr = decode_audio(req.audio_b64)
        return JSONResponse(content=run_audio_inference(arr, req.text_context or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Audio inference error")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
