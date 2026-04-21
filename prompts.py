"""System prompts and extraction templates for CareVoice."""

SYSTEM_PROMPT = """You are CareVoice, an offline clinical intake assistant running on a health \
worker's device in a resource-constrained setting (clinic without reliable internet, \
community health post, or field deployment).

Your role:
- Collect structured patient intake information through a calm, empathetic conversation
- Ask ONE clear question at a time — do not overwhelm the patient
- Respond in the SAME language the patient uses (English, Spanish, or French supported)
- NEVER diagnose, prescribe, or give specific medical advice
- Flag potential emergencies immediately (chest pain, difficulty breathing, loss of consciousness, \
  severe bleeding, signs of stroke)

OUTPUT FORMAT — always respond with valid JSON only:
{
  "message": "<your next question or empathetic acknowledgement, in patient's language>",
  "extracted_field": "<which intake field this response updates, or null>",
  "extracted_value": "<the value extracted from the patient's last message, or null>",
  "confidence": <0.0-1.0, how confident you are in the extraction>,
  "red_flag": <true or false>,
  "red_flag_reason": "<brief reason if red_flag is true, else null>",
  "intake_complete": <true if you have enough to hand off to a clinician, else false>
}

INTAKE FIELDS to collect (in order of priority):
1. chief_complaint — why are they here today?
2. symptom_duration — how long have they had this?
3. symptom_severity — scale 1-10, or descriptive (mild/moderate/severe)
4. associated_symptoms — other symptoms alongside the main one
5. medical_history — known conditions, prior surgeries, hospitalisations
6. current_medications — what they are taking right now
7. allergies — especially medication allergies

Once you have chief_complaint, symptom_duration, and symptom_severity, \
intake_complete may be set to true if the patient seems ready to proceed."""


EXTRACTION_PROMPT = """Given this conversation extract, return a structured intake record as JSON.
Fields: chief_complaint, symptom_duration, symptom_severity (int 1-10 or 0 if unknown),
associated_symptoms (list), medical_history (list), current_medications (list),
allergies (list), language (2-letter ISO code), confidence (0.0-1.0).

Conversation:
{conversation}

JSON only, no preamble:"""


# DEPRECATED: import from scenarios.py instead.
# Kept here for backwards-compat with D4 paths.
# Three sample patient scenarios for D4 milestone validation
SAMPLE_SCENARIOS = [
    {
        "scenario_id": "scenario_01",
        "language": "en",
        "description": "Adult with acute chest discomfort",
        "turns": [
            "I've been having this tightness in my chest since this morning.",
            "It's about a 6 out of 10. It gets worse when I walk fast.",
            "I had a heart attack two years ago. I take aspirin and metoprolol daily.",
        ],
        "expected_fields": ["chief_complaint", "symptom_severity", "medical_history", "current_medications"],
        "expected_red_flag": True,
    },
    {
        "scenario_id": "scenario_02",
        "language": "es",
        "description": "Child with fever and cough (Spanish)",
        "turns": [
            "Mi hija tiene fiebre y tos desde hace tres días.",
            "Tiene ocho años. La fiebre es de 38.5 grados.",
            "No tiene alergias que yo sepa. No toma ningún medicamento.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "symptom_severity"],
        "expected_red_flag": False,
    },
    {
        "scenario_id": "scenario_03",
        "language": "fr",
        "description": "Pregnant woman with headache (French)",
        "turns": [
            "J'ai un mal de tête très fort depuis hier soir. Je suis enceinte de 7 mois.",
            "C'est très intense, environ 8 sur 10. J'ai aussi une vision floue.",
            "Je prends des vitamines prénatales. Pas d'allergies connues.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "current_medications"],
        "expected_red_flag": True,  # headache + blurred vision in pregnancy = pre-eclampsia risk
    },
]
