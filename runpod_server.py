"""CareVoice inference server for RunPod L40S — Gemma 4 4B-IT.

Endpoint: POST /generate
          GET  /health
"""
import gc
import sys
import types
import json
import re
import os
import logging
import threading
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("carevoice")

# ─── Video-utils stub (not needed on RunPod, avoids import errors) ───────────
_video_stub = types.ModuleType("video_utils")
_video_stub.load_video = lambda *a, **kw: []
sys.modules.setdefault("video_utils", _video_stub)

# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/gemma4")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "true").lower() == "true"
PORT = int(os.environ.get("PORT", "8000"))

# ─── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CareVoice, a clinical intake assistant. Your role is to:
1. Collect patient information in a conversational, empathetic manner
2. Identify chief complaints and symptoms
3. Screen for red-flag symptoms that require urgent escalation
4. Summarize findings in structured JSON for clinical review

RED-FLAG symptoms requiring URGENT escalation (set urgent=true):
- Chest pain or pressure
- Difficulty breathing or shortness of breath
- Sudden severe headache ("worst headache of my life")
- Sudden weakness or numbness on one side of body
- Confusion, slurred speech, vision changes (stroke signs)
- Suicidal ideation or self-harm
- Anaphylaxis signs (throat swelling, hives + breathing difficulty)
- Uncontrolled bleeding
- Loss of consciousness

When responding, always output valid JSON in this format:
{
  "response": "<your empathetic message to the patient>",
  "extracted_info": {
    "chief_complaint": "<main reason for visit or null>",
    "symptoms": ["<list of reported symptoms>"],
    "duration": "<how long symptoms have been present or null>",
    "severity": "<mild/moderate/severe or null>",
    "medications": ["<current medications>"],
    "allergies": ["<known allergies>"],
    "urgent": <true if red-flag symptoms present, false otherwise>,
    "escalation_reason": "<reason for urgency or null>"
  },
  "intake_complete": <true if enough info collected, false if more questions needed>
}"""

# ─── Model loading ───────────────────────────────────────────────────────────
log.info(f"Loading model from {MODEL_PATH} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

# Report VRAM usage
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"VRAM: {used:.1f} GB used / {total:.1f} GB total")

log.info("Model loaded successfully.")

# Single-GPU: serialise all generate calls to avoid CUDA index assertion errors
_INFERENCE_LOCK = threading.Lock()

# ─── JSON extractor ───────────────────────────────────────────────────────────
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_RAW   = re.compile(r"(\{.*\})", re.DOTALL)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from model output."""
    for pattern in (_JSON_BLOCK, _JSON_RAW):
        m = pattern.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None

def safe_json_output(raw: str) -> Dict[str, Any]:
    """Return parsed JSON or a fallback dict with the raw text."""
    parsed = extract_json(raw)
    if parsed:
        return parsed
    # Fallback — model didn't produce valid JSON
    return {
        "response": raw.strip(),
        "extracted_info": {
            "chief_complaint": None,
            "symptoms": [],
            "duration": None,
            "severity": None,
            "medications": [],
            "allergies": [],
            "urgent": False,
            "escalation_reason": None,
        },
        "intake_complete": False,
    }

# ─── Inference ───────────────────────────────────────────────────────────────
def run_inference(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run one turn of the clinical intake conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation

    # Build prompt — apply_chat_template may return BatchEncoding or tensor
    # depending on transformers version; tokenize=False + explicit tokenize is robust
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoding = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = encoding["input_ids"]

    with _INFERENCE_LOCK:  # serialise: one generate at a time on single GPU
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=encoding.get("attention_mask"),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return safe_json_output(raw)

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="CareVoice Inference Server", version="1.0.0")


class GenerateRequest(BaseModel):
    conversation: List[Dict[str, str]]
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    response: str
    extracted_info: Dict[str, Any]
    intake_complete: bool
    raw_output: Optional[str] = None


@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
    return {"status": "ok", "model_path": MODEL_PATH, **gpu_info}


@app.post("/generate")
def generate(req: GenerateRequest):
    # Allow per-request overrides
    if req.max_new_tokens is not None:
        global MAX_NEW_TOKENS
        MAX_NEW_TOKENS = req.max_new_tokens
    if req.temperature is not None:
        global TEMPERATURE
        TEMPERATURE = req.temperature

    try:
        result = run_inference(req.conversation)
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
