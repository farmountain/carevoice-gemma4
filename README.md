# CareVoice — Offline Trimodal Clinical Intake with Gemma 4

> **Gemma 4 Good Hackathon** · Health & Sciences track  
> Text · Image · Audio — fully offline — any language — any device with 8 GB RAM

---

## What it does

A community health worker in rural Philippines sees 40 patients a day. She has no reliable internet. Before each consultation she needs: chief complaint, medications, allergies, red flags. That intake takes 10 minutes per patient × 40 patients.

CareVoice removes those barriers. It runs **entirely on the device she already owns**, processes the patient's **spoken complaint** (audio), a **photo of the wound** (image), and the **typed or transcribed text** (text) — in any language — with no cloud API, no subscription, no internet after setup.

**It does not diagnose. It does not prescribe. It prepares a structured summary for the clinician.**

---

## Modalities

| Modality | Clinical use case | Dataset |
|---|---|---|
| **Text** | Symptom collection, red-flag escalation, 16 languages | 2000+ synthetic scenarios |
| **Image** | Wound/skin triage, medication ID, X-ray review | SurgWound (CC BY-SA 4.0, 697 images) |
| **Audio** | Respiratory sound analysis (wheeze, stridor, crackle) | SPRSound (CC BY 4.0, 2,683 recordings) |

---

## Why Gemma 4 is different here

```
Gemma4ForConditionalGeneration
  vision_config:  16×16 pixel patches → 280 soft tokens per image
  audio_config:   16kHz mel-128 spectrogram → token sequence
  text_config:    262k vocab, 42 layers, 131k context window
```

Three encoders feed a single language model. A wound photo becomes 280 context tokens. A 4-second respiratory recording becomes ~60 audio tokens. Both can appear **in the same inference call**. No separate vision model. No separate ASR model.

**Critical finding**: `AutoProcessor` (not `AutoTokenizer`) is required for all inference, including text-only. The processor bundles vision, audio, and tokenizer. Using `AutoTokenizer` silently omits audio/vision preprocessing.

---

## Demo outputs

**Text — stroke red flag (English)**
```json
{
  "response": "EMERGENCY: Call 999 immediately. Face drooping + arm weakness + slurred speech are stroke signs.",
  "extracted_info": { "urgent": true, "triage_level": "red", "escalation_reason": "stroke_fast_signs" }
}
```

**Text — benign intake (Filipino)**
```
Masakit ang ulo ko nang dalawang araw, kaya ayusin ng ibuprofen.
→ { "response": "Naiintindihan ko...", "urgent": false, "triage_level": "green" }
```

**Image — wound triage**
```
SurgWound sample → { "visual_findings": { "severity_indicators": ["dehiscence"], "triage_level": "red" } }
```

**Audio — respiratory analysis**
```
SPRSound wheeze recording → { "audio_analysis": { "wheeze_present": true, "abnormal_breathing": true } }
```

---

## Quickstart

```bash
git clone https://github.com/farmountain/carevoice-gemma4
cd carevoice-gemma4/solutions/gemma_hackathon

# Local smoke test — no GPU needed (mock server)
python demo_server.py &
python eval_trimodal.py --mode http --n-text 50 --n-image 0 --n-audio 0

# Full stress test (requires running server with real model)
python eval_trimodal.py \
  --mode http \
  --url http://localhost:8000 \
  --n-text 1000 \
  --n-image 697 \
  --n-audio 200 \
  --workers 8

# CPU-only path via Ollama (4 GB RAM)
ollama pull gemma3:4b && ollama serve
# Then in notebook: infer_text_adaptive() auto-selects Ollama if HF model absent
```

---

## Architecture

```
Patient input (any modality)
        │
        ├── TEXT  ──────────► Gemma 4 text tokens  (262k vocab)
        ├── IMAGE ──────────► Vision encoder        (16×16 patches → 280 tokens/image)
        └── AUDIO ──────────► Audio encoder         (16kHz mel-128 → ~60 tokens/4s)
                                        │
                              Gemma 4 Language Model
                              42 layers · bfloat16 · ~8 GB RAM · CPU-capable
                                        │
                         ┌──────────────┴──────────────┐
                         │  Structured JSON output      │
                         │  chief_complaint             │
                         │  urgent: true/false          │
                         │  triage_level: green/yel/red │
                         │  visual_findings (image)     │
                         │  audio_analysis (audio)      │
                         └─────────────────────────────┘
```

### Input routing pattern

```python
# TEXT only
inputs = processor(text=prompt, return_tensors="pt").to(device)

# IMAGE
inputs = processor(text=prompt, images=[pil_img], return_tensors="pt").to(device)

# AUDIO — requires (array, sample_rate) tuple
inputs = processor(text=prompt, audios=[(arr, 16000)], return_tensors="pt").to(device)
```

### Nine red-flag categories

Chest pain · Respiratory distress · Stroke (FAST signs) · Thunderclap headache ·
Suicidal ideation · Anaphylaxis · Uncontrolled bleeding · Loss of consciousness · Diabetic crisis

When triggered: `urgent: true` + response generated **in the patient's current language**.

---

## Evaluation

### Text (500–2000 scenarios, combinatorial)

| Category | Metric | Target |
|---|---|---|
| Standard red-flag | Recall | ≥ 90% |
| **Adversarial RF** (downplayed phrasing) | Recall | ≥ 70% |
| Benign | False-positive rate | ≤ 15% |
| **Adversarial BN** (alarming-sounding benign) | FP rate | ≤ 30% |
| Multilingual RF | 20/20 (16 languages) | — |
| Multi-turn escalation | 5/5 | — |
| Edge cases (valid JSON) | 12/12 | — |

The evaluator covers **13 pathologies**: cardiac, stroke, respiratory, sepsis, anaphylaxis,
haemorrhage, loss_of_consciousness, obstetric, paediatric, metabolic, mental_health, toxicology, trauma.

### Image (SurgWound, 697 samples)

Triage accuracy vs surgeon ground-truth: **≥ 70%**

### Audio (SPRSound, 2,683 recordings)

Wheeze/stridor/crackle abnormal detection accuracy reported in notebook.

---

## Deployment paths

| Path | Hardware | Setup |
|---|---|---|
| **Kaggle notebook** | Free T4 GPU | `kaggle kernels push` |
| **RunPod FastAPI** | L40S 48 GB | `runpod_server_v2.py` + threading.Lock() |
| **Ollama (CPU)** | Any, 4 GB RAM | `ollama pull gemma3:4b && ollama serve` |

---

## Files

```
solutions/gemma_hackathon/
├── carevoice_trimodal_notebook.py  # Kaggle notebook source (Jupytext percent format)
├── to_ipynb.py                     # Converts .py → .ipynb for Kaggle submission
├── demo_server.py                  # Local FastAPI mock server for eval
├── eval_trimodal.py                # 1092-line stress evaluator (2000+ scenarios)
├── runpod_server_v2.py             # Production FastAPI server (RunPod)
├── writeup.md                      # Competition writeup
├── video_script.md                 # Demo video script (8 scenes)
├── submission_form.md              # Paste-ready submission form
└── kaggle_push/
    ├── notebook.ipynb              # Generated .ipynb (do not edit directly)
    └── kernel-metadata.json
```

---

## Implementation notes for other builders

1. **`AutoProcessor` is mandatory** even for text-only. Correct class: `AutoModelForImageTextToText`.
2. **`tokenize=False` + explicit `processor()` call** — avoids `BatchEncoding.shape` AttributeError in transformers ≥5.7.0.dev.
3. **`attn_implementation="eager"` on CPU** — default `sdpa` has numerical issues at bfloat16 on some CPUs.
4. **System role is supported** — Gemma 4-IT accepts `{"role": "system", ...}` directly.
5. **Audio requires `(array, sample_rate)` tuple** — `processor(audios=[(arr, 16000)], ...)`.
6. **`threading.Lock()` around `generate()`** — without it, concurrent requests crash with a CUDA index assertion.
7. **PeftConfigLike NameError fix** — monkey-patch `typing.get_type_hints` before importing transformers on `5.7.0.dev0`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*CareVoice — Offline Trimodal Clinical Intake · Powered by Gemma 4 · Gemma 4 Good Hackathon*
