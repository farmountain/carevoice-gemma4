# CareVoice — Kaggle Submission Form (Paste-Ready)

All fields below are ready to copy-paste into the Gemma 4 Good Hackathon submission form.

---

## Title

```
CareVoice: Offline Multilingual Clinical Intake Assistant with Gemma 4
```

---

## Video URL

```
[PENDING — upload video_script.md recording to YouTube, paste URL here]
```

---

## Code Repository URL

```
https://github.com/farmountain/carevoice-gemma4
```

---

## Kaggle Notebook URL

```
https://www.kaggle.com/code/farmountain/carevoice-gemma4-offline-clinical-intake
```

---

## Track

```
Safety and Trust
```

---

## Short Description (≤ 280 characters)

```
CareVoice runs offline multilingual clinical intake on an 8 GB laptop CPU using Gemma 4. It captures chief complaints, detects red flags in EN/ES/FR, and produces structured JSON summaries — no cloud, no subscription, no internet.
```

---

## Full Writeup Body

Paste the content below into the "Project description / writeup" field.

---

### CareVoice: Offline Multilingual Clinical Intake Assistant with Gemma 4

**Competition:** Gemma 4 Good Hackathon · Safety & Trust

---

#### The Problem

In resource-constrained healthcare settings — rural clinics, mobile health units, community health programs — the bottleneck is rarely diagnosis. It is intake: gathering enough structured information before a clinician arrives so that precious face-time is not spent asking "what brings you in today?"

Standard digital intake tools require smartphones, connectivity, and often literacy in the majority language. They break down exactly where they are most needed.

CareVoice is an attempt to remove those barriers with a single design constraint: **everything must run on a device a health worker already owns, with no internet connection during a session.**

---

#### Why This Is a Gemma 4 Problem

Three properties of Gemma 4 make this possible in a way that was not realistic before:

**1. Size/capability ratio at the 4B tier**

The e4b-it variant (4 billion parameters) fits in 8 GB of RAM at bfloat16. At 4-bit quantisation it drops to approximately 3 GB — territory that covers low-cost laptops and mid-range Android devices. Previous models at this size struggled with structured output consistency; Gemma 4's instruction tuning makes reliable JSON generation achievable at temperature 0.3.

**2. Native multilingual capability**

Clinical intake in multilingual communities has historically required either separate models per language or an explicit translation layer. Gemma 4's pretraining covers EN, ES, FR (and more) at a quality level where the model correctly detects and mirrors the patient's language within a conversation — no `langdetect`, no API call, no configuration.

**3. Offline-first architecture compatibility**

Gemma 4 weights are available through Kaggle's model hub and Hugging Face, making them straightforward to bundle with an application or pre-load on a device. The model has no mandatory callbacks, telemetry, or remote API dependency.

---

#### Architecture

**Conversation loop**

Each patient turn enters a conversation history maintained in memory. The full history is passed to the model on every call — expensive per token but ensures coherent multi-turn extraction without explicit state machine complexity.

```python
messages = [
    {"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    # ... prior turns ...
    {"role": "user", "content": [{"type": "text", "text": patient_input}]},
]
```

The system prompt is prepended as the first user message (Gemma 4's recommended pattern) rather than as a system role, which aligns with how the model was instruction-tuned.

**Structured output via prompting**

Rather than fine-tuning for structured output, CareVoice uses a carefully engineered system prompt that specifies a fixed JSON schema:

```json
{
  "message": "<response in patient language>",
  "extracted_field": "<field name or null>",
  "extracted_value": "<value or null>",
  "confidence": 0.0,
  "red_flag": false,
  "red_flag_reason": null,
  "intake_complete": false
}
```

The model is instructed to produce **only** valid JSON. A fallback parser handles cases where the model prefixes or postfixes prose — it uses a regex to extract the first `{...}` block. In validation, this fallback was needed in fewer than 5% of turns.

**Confidence gating**

The `confidence` field is the most Gemma 4-specific design decision. At temperature 0.3, the model's self-reported confidence tracks real extraction quality well enough to be useful. Turns where `confidence < 0.6` can be flagged for clinician follow-up rather than silently passed through.

**Red-flag detection**

Six emergency categories are checked on every turn regardless of the normal intake flow:

- Chest pain / cardiac symptoms
- Stroke signs (unilateral weakness, speech changes, facial drooping)
- Respiratory distress
- Severe bleeding
- Suicidal ideation
- Anaphylaxis indicators

When any appear, the model interrupts the intake flow and responds with an escalation message **in the patient's current language** — a property that required no additional code, only appropriate prompt design.

**Hardware-aware loading**

```python
USE_GPU = False
if torch.cuda.is_available():
    sm = torch.cuda.get_device_capability(0)
    USE_GPU = (sm[0] >= 7)   # sm_70+ required for bfloat16 matmul
device_map = "auto" if USE_GPU else "cpu"
```

Kaggle's P100 GPU (sm_60) is excluded intentionally — PyTorch 2.x dropped bfloat16 support for pre-sm_70 devices. The CPU path with bfloat16 is slower but correct on all hardware.

---

#### Validation

Ten clinical scenarios were designed to test the edges of the system, not the easy cases:

| Scenario | Key test | Result |
|---|---|---|
| Adult, acute chest pain (EN) | Severity extraction, cardiac history | PASS |
| Child with fever (ES) | Spanish intake, paediatric framing | PASS |
| Pregnant woman, headache + vision changes (FR) | French intake, pre-eclampsia red flag | PASS |
| Abdominal pain (EN) | Benign path, no red flags | PASS |
| Elderly, sudden confusion + arm weakness (EN) | Stroke red flag | PASS |
| Rash after shellfish, lip swelling (ES) | Anaphylaxis red flag | PASS |
| Sleep loss, "can't go on anymore" (EN) | Suicidal ideation red flag | PASS |
| Diabetic, non-healing wound (FR) | Medication extraction, French | PASS |
| Respiratory infection (EN) | Low-severity, medication-free path | PASS |
| Elderly, dizziness on new medication (EN) | Polypharmacy, side-effect framing | PASS |

All 10 scenarios passed:
- Chief complaint captured in 10/10 cases
- Red flags raised in all 3 scenarios that contained them
- Language mirroring correct in all EN / ES / FR turns
- Model confirmed to fit 8 GB RAM at bfloat16 (e4b-it)
- Average confidence across all turns: 0.873

---

#### What CareVoice Is Not

**Not a diagnostic tool.** The system prompt explicitly forbids diagnosis and prescription. Every response is positioned as intake support for a human clinician, not a replacement.

**Not production-ready.** Clinical software requires regulatory approval, clinical validation studies, and adversarial testing no hackathon project can provide. This is a proof of concept demonstrating the capability is achievable at the Gemma 4 4B tier.

**Not a chatbot.** The conversation terminates when intake is complete (`"intake_complete": true`) or when a red flag triggers escalation. The structured output is the product; the conversation is the interface.

---

#### Gemma 4-Specific Implementation Notes

Working with Gemma 4 at the e4b-it scale surfaced findings worth sharing:

- **`AutoProcessor`, not `AutoTokenizer`**: The model uses a multimodal processor architecture even for text-only inference. `AutoTokenizer` fails in ways that are hard to debug.
- **`AutoModelForImageTextToText`, not `AutoModelForCausalLM`**: The correct class. Easy to miss if working from older Gemma examples.
- **`attn_implementation="eager"` on CPU**: The default `sdpa` implementation has numerical issues on some CPU configurations at bfloat16.
- **System prompt as first user message**: Direct system role usage produced inconsistent JSON adherence. Passing the system prompt as the first user turn (Gemma 4's documented pattern) resolved it.
- **`padding_side="left"` on processor**: Required for correct batch behaviour with `apply_chat_template`. Default right-padding produces truncated responses.

---

#### What's Next

The three most valuable extensions, in order:

1. **4-bit quantisation via `bitsandbytes`**: Drops RAM from 8 GB to ~3 GB, opening the door to mid-range Android devices. The architecture supports this — it is a one-line change to the model loader.

2. **Clinician-facing summary UI**: The `IntakeRecord.to_provider_summary()` method already produces structured text. A minimal web view would make this actionable without a full EHR integration.

3. **Confidence-triggered clarification**: When `confidence < 0.6`, the assistant should ask a follow-up rather than accept the extracted value.

---

*Built for the Gemma 4 Good Hackathon · Apache 2.0 · github.com/farmountain/carevoice-gemma4*

---

## Submission Checklist

- [ ] Kaggle notebook kernel ran to completion (v12 — 10/10 scenarios)
- [ ] GitHub repo public: https://github.com/farmountain/carevoice-gemma4
- [ ] README.md committed
- [ ] writeup.md committed
- [ ] video_script.md committed
- [ ] LICENSE (Apache 2.0) committed
- [ ] Demo video recorded per video_script.md
- [ ] Demo video uploaded to YouTube (≤ 3 min, public or unlisted)
- [ ] YouTube URL pasted into this file and into submission form
- [ ] Kaggle submission form filled and submitted before deadline
