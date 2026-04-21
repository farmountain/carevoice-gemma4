# CareVoice Demo Video Script
**Target length:** 2:30 – 3:00  
**Format:** Screen recording + voiceover. No talking head required.  
**Tone:** Calm, clinical, grounded. Not a tech demo — a use-case demo.

---

## SCENE 1 — The Problem (0:00 – 0:25)

**[SCREEN: Black. Text fades in.]**

> *"A community health worker in a rural clinic sees 40 patients a day."*  
> *"She has no reliable internet. Her laptop has 8 GB of RAM."*  
> *"Before each consultation, she needs: chief complaint, medications, allergies, red flags."*  
> *"That intake takes 10 minutes. Multiplied by 40 patients."*

**[SCREEN: Text fades out. CareVoice terminal opens.]**

**VOICEOVER:**
> "CareVoice runs entirely on her device. No cloud. No subscription. No internet after setup. Just Gemma 4."

---

## SCENE 2 — English Intake: Chest Pain (0:25 – 0:55)

**[SCREEN: Terminal running demo_cli.py. Clean, readable font.]**

**VOICEOVER:**
> "A patient walks in speaking English. He has chest tightness."

**[SCREEN: Patient input appears:]**
```
Patient: I've been having this tightness in my chest since this morning.
```

**[SCREEN: CareVoice responds:]**
```
CareVoice: {"message": "I'm sorry to hear that. On a scale of 1 to 10,
            how would you rate the discomfort right now?",
            "extracted_field": "chief_complaint",
            "extracted_value": "chest tightness",
            "confidence": 0.94,
            "red_flag": false}
```

**VOICEOVER:**
> "CareVoice captures the chief complaint immediately and asks the right next question — severity. Every response is structured JSON, ready for a clinician's dashboard."

---

## SCENE 3 — Spanish Intake (0:55 – 1:20)

**[SCREEN: New scenario. Patient input in Spanish.]**

```
Patient: Mi hija tiene fiebre y tos desde hace tres dias.
```

**[SCREEN: CareVoice responds in Spanish:]**
```
CareVoice: {"message": "Entiendo. ¿Cuántos años tiene su hija y
            tiene dificultad para respirar?",
            "extracted_field": "chief_complaint",
            "extracted_value": "fiebre y tos",
            "confidence": 0.91,
            "red_flag": false}
```

**VOICEOVER:**
> "The patient switches to Spanish. CareVoice detects it automatically and responds in kind — no configuration, no language selection. This is Gemma 4's multilingual pretraining at work."

---

## SCENE 4 — Emergency Red Flag (1:20 – 1:50)

**[SCREEN: New scenario. Setup text: "Stroke signs".]**

```
Patient: My father is 78. He suddenly got confused and fell.
         He can't move his left arm and his speech is slurred.
```

**[SCREEN: CareVoice response with red_flag highlighted:]**
```
CareVoice: {"message": "This sounds like a medical emergency.
            Please call emergency services immediately.
            Do not wait.",
            "red_flag": true,
            "red_flag_reason": "Stroke signs: unilateral weakness +
                                slurred speech + sudden confusion",
            "confidence": 0.98}
```

**VOICEOVER:**
> "When stroke signs appear, CareVoice interrupts the intake entirely. It escalates in the patient's language — because a missed red flag is not a UX problem. It's a patient safety problem."

---

## SCENE 5 — Provider Summary (1:50 – 2:10)

**[SCREEN: End of scenario. Provider summary prints:]**

```
--- Provider Summary ---
CHIEF COMPLAINT:  chest tightness since this morning
SEVERITY:         6/10, worse on exertion
MEDICAL HISTORY:  heart attack (2 years ago)
MEDICATIONS:      aspirin, metoprolol
RED FLAGS:        none
```

**VOICEOVER:**
> "After the conversation, a clean structured summary is ready for the clinician. No typing. No transcription. Just the facts, in the format a provider actually needs."

---

## SCENE 6 — Hardware Proof (2:10 – 2:30)

**[SCREEN: System info from the Kaggle notebook output:]**

```
Model:   gemma-4-e4b-it  (4 billion parameters)
Device:  CPU  (no GPU required)
RAM:     bfloat16 = 8 GB  |  4-bit = ~3 GB
```

**VOICEOVER:**
> "This runs on a standard laptop CPU. No GPU. No cloud API. Gemma 4's 4-billion parameter model fits in 8 gigabytes — the same RAM as a basic school computer. At 4-bit quantisation, it drops to 3."

---

## SCENE 7 — Close (2:30 – 2:50)

**[SCREEN: GitHub URL fades in: github.com/farmountain/carevoice-gemma4]**  
**[SCREEN: Kaggle notebook link fades in below.]**

**VOICEOVER:**
> "CareVoice is open source under Apache 2.0. The full code, Kaggle notebook, and technical writeup are linked below. Built for the Gemma 4 Good Hackathon — Safety and Trust track."

**[SCREEN: Fade to black. Text:]**
> *CareVoice — Offline Clinical Intake*  
> *Powered by Gemma 4*

---

## Recording Notes

- **Screen recorder:** OBS Studio (free) or Windows Game Bar (`Win + G`)
- **Terminal font:** Consolas 16pt or JetBrains Mono, white on dark background
- **Voiceover:** Record separately, sync in post. Speak at ~130 words/min.
- **Total voiceover word count:** ~310 words → fits in 2:30
- **No face cam required** — screen + voice is sufficient per competition rules
- **Upload:** YouTube (unlisted or public), paste link in Kaggle submission form

---

## Rubric coverage per scene

| Criterion | Scenes |
|---|---|
| Innovation (30%) | 3 (auto language), 4 (red flag interruption), 5 (structured output) |
| Impact (30%) | 1 (problem framing), 4 (patient safety) |
| Technical Execution (25%) | 2 (JSON output), 5 (provider summary) |
| Accessibility (15%) | 6 (CPU, 8 GB RAM proof) |
