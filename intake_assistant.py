"""CareVoice — Offline Clinical Intake Assistant using Gemma 4.

Designed for the Gemma 4 Good Hackathon (gemma-4-good-hackathon).
Runs fully offline after model download. Targets 8GB RAM consumer hardware
via 4-bit quantisation of Gemma 4 7B-IT.

Usage:
    # Real inference (requires GPU + transformers):
    assistant = IntakeAssistant.load(model_id="google/gemma-4-7b-it")

    # Mock mode (no GPU required, deterministic — for D4 local validation):
    assistant = IntakeAssistant.mock()

    result = assistant.run_scenario(scenario)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .prompts import SYSTEM_PROMPT
from .scenarios import SAMPLE_SCENARIOS


# ── Intake record data model ──────────────────────────────────────────────────

@dataclass
class IntakeRecord:
    chief_complaint: str = ""
    symptom_duration: str = ""
    symptom_severity: int = 0       # 0 = unknown, 1-10 scale
    associated_symptoms: list[str] = field(default_factory=list)
    medical_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    language: str = "en"
    red_flags: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0

    def is_ready_for_clinician(self) -> bool:
        return bool(self.chief_complaint and self.symptom_duration)

    def to_provider_summary(self) -> str:
        lines = [
            f"CHIEF COMPLAINT: {self.chief_complaint or '(not captured)'}",
            f"DURATION: {self.symptom_duration or '(not captured)'}",
            f"SEVERITY: {self.symptom_severity}/10" if self.symptom_severity else "SEVERITY: not rated",
        ]
        if self.associated_symptoms:
            lines.append(f"ASSOCIATED: {', '.join(self.associated_symptoms)}")
        if self.medical_history:
            lines.append(f"PMH: {'; '.join(self.medical_history)}")
        if self.current_medications:
            lines.append(f"MEDICATIONS: {', '.join(self.current_medications)}")
        if self.allergies:
            lines.append(f"ALLERGIES: {', '.join(self.allergies)}")
        if self.red_flags:
            lines.append(f"[!] RED FLAGS: {'; '.join(self.red_flags)}")
        return "\n".join(lines)


@dataclass
class TurnResult:
    turn: int
    patient_input: str
    assistant_message: str
    extracted_field: str | None
    extracted_value: Any
    confidence: float
    red_flag: bool
    red_flag_reason: str | None
    intake_complete: bool
    raw_response: str


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    turns: list[TurnResult]
    final_record: IntakeRecord
    passed_d4_criterion: bool
    notes: list[str] = field(default_factory=list)


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """Extract the first JSON object from a model response string."""
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Return a safe fallback
    return {
        "message": raw,
        "extracted_field": None,
        "extracted_value": None,
        "confidence": 0.0,
        "red_flag": False,
        "red_flag_reason": None,
        "intake_complete": False,
    }


def _apply_extraction(record: IntakeRecord, field_name: str | None, value: Any) -> None:
    if not field_name or value is None:
        return
    field_name = field_name.lower().strip()
    if field_name == "chief_complaint":
        record.chief_complaint = str(value)
    elif field_name == "symptom_duration":
        record.symptom_duration = str(value)
    elif field_name == "symptom_severity":
        try:
            record.symptom_severity = int(value)
        except (ValueError, TypeError):
            pass
    elif field_name in ("associated_symptoms", "associated"):
        vals = value if isinstance(value, list) else [str(value)]
        record.associated_symptoms.extend(vals)
    elif field_name in ("medical_history", "history"):
        vals = value if isinstance(value, list) else [str(value)]
        record.medical_history.extend(vals)
    elif field_name in ("current_medications", "medications"):
        vals = value if isinstance(value, list) else [str(value)]
        record.current_medications.extend(vals)
    elif field_name == "allergies":
        vals = value if isinstance(value, list) else [str(value)]
        record.allergies.extend(vals)
    elif field_name == "language":
        record.language = str(value)


# ── Mock model for local D4 validation ───────────────────────────────────────

_DEFAULT_MOCK_TURN = {
    "message": "Thank you. Can you tell me more about when this started?",
    "extracted_field": None,
    "extracted_value": None,
    "confidence": 0.75,
    "red_flag": False,
    "red_flag_reason": None,
    "intake_complete": False,
}

_MOCK_RESPONSES: dict[str, list[dict]] = {
    "scenario_01": [
        {
            "message": "I'm sorry to hear that. Can you tell me how severe the tightness feels on a scale of 1 to 10?",
            "extracted_field": "chief_complaint",
            "extracted_value": "chest tightness since this morning",
            "confidence": 0.92,
            "red_flag": True,
            "red_flag_reason": "Chest tightness may indicate cardiac event — must be assessed urgently",
            "intake_complete": False,
        },
        {
            "message": "Thank you. Does the tightness radiate to your arm, jaw, or shoulder?",
            "extracted_field": "symptom_severity",
            "extracted_value": 6,
            "confidence": 0.95,
            "red_flag": True,
            "red_flag_reason": "Exertional chest pain warrants urgent cardiac assessment",
            "intake_complete": False,
        },
        {
            "message": "Given your history and current symptoms, a clinician should assess you immediately. I have your intake ready.",
            "extracted_field": "medical_history",
            "extracted_value": ["prior myocardial infarction (2 years ago)"],
            "confidence": 0.90,
            "red_flag": True,
            "red_flag_reason": "Prior MI + exertional chest pain = high-risk presentation",
            "intake_complete": True,
        },
    ],
    "scenario_02": [
        {
            "message": "Entiendo. ¿Cuántos años tiene su hija y tiene otros síntomas además de la fiebre y la tos?",
            "extracted_field": "chief_complaint",
            "extracted_value": "fiebre y tos por tres días",
            "confidence": 0.93,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Gracias. ¿La tos es seca o con flema? ¿Tiene dificultad para respirar?",
            "extracted_field": "symptom_severity",
            "extracted_value": 4,
            "confidence": 0.80,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Entendido. Tengo la información necesaria para el médico. ¿Hay algo más que quiera agregar?",
            "extracted_field": "allergies",
            "extracted_value": [],
            "confidence": 0.88,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": True,
        },
    ],
    "scenario_03": [
        {
            "message": "Je comprends. Depuis combien de temps exactement avez-vous ce mal de tête ?",
            "extracted_field": "chief_complaint",
            "extracted_value": "mal de tête intense depuis hier soir, grossesse 7 mois",
            "confidence": 0.94,
            "red_flag": True,
            "red_flag_reason": "Céphalée sévère en fin de grossesse — risque de pré-éclampsie",
            "intake_complete": False,
        },
        {
            "message": "Une vision floue avec des maux de tête sévères pendant la grossesse nécessite une attention médicale immédiate. Je prépare votre fiche pour le médecin.",
            "extracted_field": "symptom_severity",
            "extracted_value": 8,
            "confidence": 0.96,
            "red_flag": True,
            "red_flag_reason": "Céphalée 8/10 + vision floue à 7 mois = alerte pré-éclampsie",
            "intake_complete": False,
        },
        {
            "message": "Votre fiche est prête. Le médecin va vous voir immédiatement. Restez calme.",
            "extracted_field": "current_medications",
            "extracted_value": ["vitamines prénatales"],
            "confidence": 0.89,
            "red_flag": True,
            "red_flag_reason": "Urgence obstétricale potentielle",
            "intake_complete": True,
        },
    ],
    # ── D12 new scenarios ──────────────────────────────────────────────────
    "scenario_04": [
        {
            "message": "I'm sorry to hear that. Is the pain constant or does it come and go?",
            "extracted_field": "chief_complaint",
            "extracted_value": "stomach pain and diarrhea for two days",
            "confidence": 0.91,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Have you noticed any blood in the diarrhea, or do you have a fever?",
            "extracted_field": "symptom_severity",
            "extracted_value": 4,
            "confidence": 0.87,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Thank you. I have enough to prepare your intake form. Is there anything else?",
            "extracted_field": "medical_history",
            "extracted_value": [],
            "confidence": 0.83,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": True,
        },
    ],
    "scenario_05": [
        {
            "message": "This sounds very serious. Is your father conscious and breathing normally right now?",
            "extracted_field": "chief_complaint",
            "extracted_value": "sudden confusion and fall, left arm weakness, slurred speech in 78-year-old",
            "confidence": 0.95,
            "red_flag": True,
            "red_flag_reason": "Sudden confusion + unilateral weakness + slurred speech = stroke (FAST criteria met)",
            "intake_complete": False,
        },
        {
            "message": "Please call emergency services immediately — these are signs of a stroke. While waiting, keep him still and note the exact time symptoms started.",
            "extracted_field": "associated_symptoms",
            "extracted_value": ["left arm weakness", "slurred speech"],
            "confidence": 0.97,
            "red_flag": True,
            "red_flag_reason": "FAST-positive presentation — time-critical for thrombolysis",
            "intake_complete": False,
        },
        {
            "message": "I have alerted the clinical team. Penicillin allergy noted. Stroke protocol activated.",
            "extracted_field": "current_medications",
            "extracted_value": ["lisinopril"],
            "confidence": 0.92,
            "red_flag": True,
            "red_flag_reason": "Acute stroke — requires immediate intervention",
            "intake_complete": True,
        },
    ],
    "scenario_06": [
        {
            "message": "¿Tiene dificultad para respirar o siente que la garganta se le cierra?",
            "extracted_field": "chief_complaint",
            "extracted_value": "erupción en brazos y pecho con picazón, después de comer mariscos",
            "confidence": 0.93,
            "red_flag": True,
            "red_flag_reason": "Erupción + hinchazón labial tras ingesta de alimento = posible anafilaxia",
            "intake_complete": False,
        },
        {
            "message": "Necesita evaluación médica urgente. ¿Tiene EpiPen o algún antihistamínico disponible?",
            "extracted_field": "associated_symptoms",
            "extracted_value": ["hinchazón en labios"],
            "confidence": 0.94,
            "red_flag": True,
            "red_flag_reason": "Angioedema + urticaria = riesgo de anafilaxia",
            "intake_complete": False,
        },
        {
            "message": "El médico le verá de inmediato. Su ficha está lista.",
            "extracted_field": "allergies",
            "extracted_value": ["mariscos (probable)"],
            "confidence": 0.88,
            "red_flag": True,
            "red_flag_reason": "Reacción alérgica severa — monitoreo urgente",
            "intake_complete": True,
        },
    ],
    "scenario_07": [
        {
            "message": "I hear you, and I want you to know you're not alone. Are you having thoughts of harming yourself or ending your life?",
            "extracted_field": "chief_complaint",
            "extracted_value": "insomnia, hopelessness, thoughts of not going on (3 weeks)",
            "confidence": 0.89,
            "red_flag": True,
            "red_flag_reason": "Suicidal ideation language — immediate safety assessment required",
            "intake_complete": False,
        },
        {
            "message": "Thank you for trusting me with this. A mental health clinician will speak with you right away. You are safe here.",
            "extracted_field": "symptom_duration",
            "extracted_value": "three weeks",
            "confidence": 0.87,
            "red_flag": True,
            "red_flag_reason": "Prior episodes + current ideation + medication discontinuation = high risk",
            "intake_complete": False,
        },
        {
            "message": "I've flagged this as urgent. The clinician is on their way. Please stay with me.",
            "extracted_field": "current_medications",
            "extracted_value": ["antidepressants (discontinued 1 month ago)"],
            "confidence": 0.90,
            "red_flag": True,
            "red_flag_reason": "Abrupt antidepressant discontinuation + active ideation",
            "intake_complete": True,
        },
    ],
    "scenario_08": [
        {
            "message": "Je comprends. Est-ce que la plaie est rouge, chaude ou a-t-elle du pus ?",
            "extracted_field": "chief_complaint",
            "extracted_value": "plaie au pied droit non cicatrisante depuis 2 semaines, odeur",
            "confidence": 0.92,
            "red_flag": True,
            "red_flag_reason": "Pied diabétique infecté — risque sepsis/gangrène",
            "intake_complete": False,
        },
        {
            "message": "Une plaie infectée chez un diabétique nécessite une évaluation urgente. Avez-vous de la fièvre ?",
            "extracted_field": "medical_history",
            "extracted_value": ["diabète type 2 (10 ans)"],
            "confidence": 0.94,
            "red_flag": True,
            "red_flag_reason": "Diabète + infection du pied = risque d'amputation",
            "intake_complete": False,
        },
        {
            "message": "Allergie à la pénicilline bien notée. Le médecin va vous examiner immédiatement.",
            "extracted_field": "allergies",
            "extracted_value": ["pénicilline"],
            "confidence": 0.96,
            "red_flag": True,
            "red_flag_reason": "Pied diabétique infecté — urgence chirurgicale potentielle",
            "intake_complete": True,
        },
    ],
    "scenario_09": [
        {
            "message": "Sorry to hear that. Do you have any difficulty swallowing or breathing?",
            "extracted_field": "chief_complaint",
            "extracted_value": "sore throat, runny nose, low fever for 4 days",
            "confidence": 0.90,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Have you had any contact with someone who was recently ill?",
            "extracted_field": "symptom_severity",
            "extracted_value": 3,
            "confidence": 0.85,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Thank you. Your intake form is ready for the clinician.",
            "extracted_field": "medical_history",
            "extracted_value": [],
            "confidence": 0.88,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": True,
        },
    ],
    "scenario_10": [
        {
            "message": "That's important to report. Is the dizziness constant or only when changing position?",
            "extracted_field": "chief_complaint",
            "extracted_value": "dizziness after starting ramipril, worst on standing, near-fall",
            "confidence": 0.91,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "Dizziness on standing can be a side effect of blood pressure medication. When did you take your last dose?",
            "extracted_field": "symptom_duration",
            "extracted_value": "since starting ramipril last week",
            "confidence": 0.89,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": False,
        },
        {
            "message": "I've noted all your medications. The clinician will review your blood pressure and medication dosing.",
            "extracted_field": "current_medications",
            "extracted_value": ["amlodipine", "atorvastatin", "ramipril"],
            "confidence": 0.93,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": True,
        },
    ],
}


class _MockModel:
    """Deterministic mock — no GPU needed, returns pre-scripted responses."""

    def generate(self, conversation: list[dict], scenario_id: str, turn: int) -> str:
        responses = _MOCK_RESPONSES.get(scenario_id, [])
        if turn < len(responses):
            return json.dumps(responses[turn])
        return json.dumps({
            "message": "Thank you for the information. Is there anything else you'd like to add?",
            "extracted_field": None,
            "extracted_value": None,
            "confidence": 0.5,
            "red_flag": False,
            "red_flag_reason": None,
            "intake_complete": True,
        })


# ── Real model wrapper ────────────────────────────────────────────────────────

class _GemmaModel:
    """Wraps a loaded Gemma 4 model for clinical intake conversations."""

    def __init__(self, model, tokenizer, device: str):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def generate(self, conversation: list[dict], scenario_id: str, turn: int) -> str:
        # Build messages for chat template
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation)

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main assistant class ──────────────────────────────────────────────────────

class IntakeAssistant:
    def __init__(self, model):
        self._model = model

    @classmethod
    def mock(cls) -> "IntakeAssistant":
        """Instantiate with mock model — no GPU or model download required."""
        return cls(_MockModel())

    @classmethod
    def load(
        cls,
        model_id: str = "google/gemma-4-7b-it",
        use_4bit: bool = True,
        kaggle_model_path: str | None = None,
    ) -> "IntakeAssistant":
        """Load real Gemma 4 model. Requires transformers + torch + bitsandbytes."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        load_path = kaggle_model_path or model_id
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {load_path} on {device} (4bit={use_4bit})...")

        bnb_config = None
        if use_4bit and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        model.eval()

        return cls(_GemmaModel(model, tokenizer, device))

    def run_scenario(self, scenario: dict) -> ScenarioResult:
        """Run a full scenario through the intake conversation."""
        scenario_id = scenario["scenario_id"]
        patient_turns = scenario["turns"]

        record = IntakeRecord(language=scenario.get("language", "en"))
        conversation: list[dict] = []
        turn_results: list[TurnResult] = []

        for i, patient_input in enumerate(patient_turns):
            # Add patient message to conversation
            conversation.append({"role": "user", "content": patient_input})

            # Generate model response
            raw = self._model.generate(conversation, scenario_id, i)
            parsed = _parse_json_response(raw)

            # Update record
            _apply_extraction(record, parsed.get("extracted_field"), parsed.get("extracted_value"))
            if parsed.get("red_flag") and parsed.get("red_flag_reason"):
                if parsed["red_flag_reason"] not in record.red_flags:
                    record.red_flags.append(parsed["red_flag_reason"])

            assistant_msg = parsed.get("message", "")
            conversation.append({"role": "assistant", "content": assistant_msg})

            turn_results.append(TurnResult(
                turn=i,
                patient_input=patient_input,
                assistant_message=assistant_msg,
                extracted_field=parsed.get("extracted_field"),
                extracted_value=parsed.get("extracted_value"),
                confidence=float(parsed.get("confidence", 0.0)),
                red_flag=bool(parsed.get("red_flag", False)),
                red_flag_reason=parsed.get("red_flag_reason"),
                intake_complete=bool(parsed.get("intake_complete", False)),
                raw_response=raw,
            ))

        # Compute overall confidence
        confidences = [t.confidence for t in turn_results if t.confidence > 0]
        record.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # D4 success criterion: structured output produced, no internet calls, no crash
        passed = (
            bool(record.chief_complaint)               # captured chief complaint
            and len(turn_results) == len(patient_turns) # all turns completed
            and all(t.assistant_message for t in turn_results)  # all responses non-empty
        )

        notes = []
        if not record.chief_complaint:
            notes.append("WARN: chief_complaint not extracted")
        if not record.symptom_duration:
            notes.append("NOTE: symptom_duration not extracted (not required for D4)")
        expected_red_flag = scenario.get("expected_red_flag", False)
        actual_red_flag = bool(record.red_flags)
        if expected_red_flag and not actual_red_flag:
            notes.append("WARN: expected red flag not triggered")
        elif not expected_red_flag and actual_red_flag:
            notes.append("NOTE: unexpected red flag triggered")

        return ScenarioResult(
            scenario_id=scenario_id,
            description=scenario["description"],
            turns=turn_results,
            final_record=record,
            passed_d4_criterion=passed,
            notes=notes,
        )
