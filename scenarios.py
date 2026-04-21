"""All patient scenarios for CareVoice validation.

D4  used scenarios 1-3.
D12 uses all 10.

Design goals for the full set:
- Coverage: acute/chronic/mental-health/paediatric/OB/geriatric
- Languages: English (6), Spanish (2), French (2)
- Red-flag rate: ~40% to validate safety logic without over-triggering
- Variety: routine presentations alongside true emergencies
"""

SAMPLE_SCENARIOS = [
    # ── Previously validated (D4) ──────────────────────────────────────────
    {
        "scenario_id": "scenario_01",
        "language": "en",
        "description": "Adult with acute chest discomfort (cardiac red flag)",
        "category": "acute_cardiac",
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
        "description": "Child with fever and cough (Spanish, routine)",
        "category": "paediatric_respiratory",
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
        "description": "Pregnant woman with headache and visual disturbance (pre-eclampsia risk)",
        "category": "obstetric_emergency",
        "turns": [
            "J'ai un mal de tête très fort depuis hier soir. Je suis enceinte de 7 mois.",
            "C'est très intense, environ 8 sur 10. J'ai aussi une vision floue.",
            "Je prends des vitamines prénatales. Pas d'allergies connues.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "current_medications"],
        "expected_red_flag": True,
    },

    # ── New scenarios (D12) ────────────────────────────────────────────────
    {
        "scenario_id": "scenario_04",
        "language": "en",
        "description": "Adult with abdominal pain (routine GI)",
        "category": "gastrointestinal",
        "turns": [
            "I've had stomach pain and diarrhea for two days now.",
            "The pain is around a 4. It's crampy, mostly in my lower belly.",
            "I don't have any chronic conditions. No regular medications.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "symptom_severity"],
        "expected_red_flag": False,
    },
    {
        "scenario_id": "scenario_05",
        "language": "en",
        "description": "Elderly patient with sudden confusion and fall (stroke red flag)",
        "category": "neurological_emergency",
        "turns": [
            "My father is 78. He suddenly got confused this morning and fell.",
            "He can't seem to move his left arm properly and his speech is slurred.",
            "He has high blood pressure and takes lisinopril. No known allergies.",
        ],
        "expected_fields": ["chief_complaint", "medical_history", "current_medications"],
        "expected_red_flag": True,
    },
    {
        "scenario_id": "scenario_06",
        "language": "es",
        "description": "Adult with skin rash and itching (Spanish, routine allergic)",
        "category": "dermatology",
        "turns": [
            "Tengo una erupción en los brazos y el pecho desde ayer. Me pica mucho.",
            "Empezó después de que comí mariscos en la noche. Tengo hinchazón en los labios.",
            "Nunca me había pasado esto. No tomo medicamentos.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "medical_history"],
        "expected_red_flag": True,   # lip swelling + rash after food = potential anaphylaxis
    },
    {
        "scenario_id": "scenario_07",
        "language": "en",
        "description": "Adult presenting with mental health crisis (sensitive, safety protocol)",
        "category": "mental_health",
        "turns": [
            "I haven't been sleeping and I feel like I can't go on anymore.",
            "I've been feeling this way for about three weeks. I've had these thoughts before.",
            "I'm not taking anything right now. I stopped my antidepressants a month ago.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "current_medications"],
        "expected_red_flag": True,   # suicidal ideation language requires immediate escalation
    },
    {
        "scenario_id": "scenario_08",
        "language": "fr",
        "description": "Diabetic adult with non-healing foot wound (chronic disease management)",
        "category": "chronic_diabetes",
        "turns": [
            "J'ai une plaie au pied droit qui ne guérit pas depuis deux semaines.",
            "Je suis diabétique de type 2 depuis dix ans. La blessure commence à sentir mauvais.",
            "Je prends de la metformine et de l'insuline. Je suis allergique à la pénicilline.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "medical_history",
                             "current_medications", "allergies"],
        "expected_red_flag": True,   # infected diabetic foot = risk of sepsis/amputation
    },
    {
        "scenario_id": "scenario_09",
        "language": "en",
        "description": "Adult with respiratory infection (routine upper respiratory)",
        "category": "respiratory_infection",
        "turns": [
            "I have a sore throat, runny nose, and a low fever for the past four days.",
            "My temperature was 37.8 this morning. The throat pain is about a 3 out of 10.",
            "I'm generally healthy. I'm not on any medications. No drug allergies.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "symptom_severity"],
        "expected_red_flag": False,
    },
    {
        "scenario_id": "scenario_10",
        "language": "en",
        "description": "Elderly patient asking about medication side effects (medication safety)",
        "category": "medication_safety",
        "turns": [
            "I started a new blood pressure tablet last week and I've been feeling dizzy.",
            "The dizziness is worst when I stand up. I almost fell yesterday.",
            "I'm 72. I take amlodipine, atorvastatin, and now this new one — ramipril.",
        ],
        "expected_fields": ["chief_complaint", "symptom_duration", "current_medications"],
        "expected_red_flag": False,  # orthostatic hypotension — important but not emergency
    },
]

# Subset used in D4 (first 3 only)
D4_SCENARIOS = SAMPLE_SCENARIOS[:3]
# All 10 for D12
D12_SCENARIOS = SAMPLE_SCENARIOS
