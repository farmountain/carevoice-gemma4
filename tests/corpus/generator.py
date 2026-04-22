"""
CareVoice Clinical Intake — Synthetic Scenario Corpus Generator
================================================================
Generates deterministic, fully programmatic test scenarios for the
CareVoice AI clinical intake assistant (Gemma 4 evaluation harness).

No external data or API calls — purely Python stdlib (random, copy).

Usage
-----
    from tests.corpus.generator import generate_corpus, EXTENDED_CORPUS

    scenarios = generate_corpus(n=500, seed=42)   # reproducible
    print(len(EXTENDED_CORPUS))                   # 500
"""

import copy
import random
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Red-flag turn templates
# Each entry: (category, description_template, turn_sets, base_fields)
# turn_sets is a list of lists; each inner list is a 2-4 turn sequence.
# ---------------------------------------------------------------------------

RED_FLAG_TEMPLATES: List[Dict[str, Any]] = [
    # ── cardiac ──────────────────────────────────────────────────────────────
    {
        "category": "acute_cardiac",
        "red_flag_category": "cardiac",
        "description": "Adult with acute chest discomfort",
        "turn_sets": [
            [
                "I've been having this really tight feeling in my chest for the past hour.",
                "It's about a 7 out of 10 for pain. It started while I was just sitting.",
                "I had a heart attack two years ago and I'm on aspirin and a beta-blocker.",
                "My left arm feels a bit numb too.",
            ],
            [
                "My chest feels like someone is squeezing it. It's been going on for maybe 45 minutes.",
                "The pain is pretty severe, maybe an 8 out of 10.",
                "I'm also sweating a lot even though I haven't been moving around.",
            ],
            [
                "I have chest pain that radiates up to my jaw. It started about 20 minutes ago.",
                "The pain is crushing, maybe a 9 out of 10. I feel very scared.",
                "I have high blood pressure and take lisinopril. No known drug allergies.",
            ],
            [
                "I keep getting these palpitations and then I feel dizzy and lightheaded.",
                "It's happened three times today. Each episode lasts a few minutes.",
                "I feel like my heart is racing then suddenly pounding irregularly.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "medical_history", "current_medications"],
    },
    # ── stroke ───────────────────────────────────────────────────────────────
    {
        "category": "acute_stroke",
        "red_flag_category": "stroke",
        "description": "Patient with sudden neurological symptoms",
        "turn_sets": [
            [
                "My right arm suddenly went weak about 30 minutes ago and I can barely lift it.",
                "My speech also started sounding slurred. My wife says I sound strange.",
                "I'm 68, I have atrial fibrillation and take warfarin.",
            ],
            [
                "I suddenly got very confused and then I fell down. I'm not sure what happened.",
                "My daughter is helping me right now. She says one side of my face looks droopy.",
                "This came on very suddenly, maybe 15 minutes ago. No pain.",
            ],
            [
                "The left side of my face is drooping and I have the worst headache of my life.",
                "It started maybe 20 minutes ago, completely out of nowhere.",
                "I have never had a headache this severe before. It's a 10 out of 10.",
            ],
            [
                "I woke up and suddenly could not understand what people were saying to me.",
                "I am trying to speak but the words are not coming out right.",
                "My right leg also feels very weak. I am 72 years old.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "medical_history", "current_medications"],
    },
    # ── respiratory_distress ─────────────────────────────────────────────────
    {
        "category": "acute_respiratory_distress",
        "red_flag_category": "respiratory_distress",
        "description": "Patient with severe breathing difficulty",
        "turn_sets": [
            [
                "I can barely breathe. I've had asthma for years but my inhaler is not helping at all.",
                "I've used my rescue inhaler four times in the last hour. My lips feel bluish.",
                "I'm really scared. I can't finish a full sentence without stopping for air.",
            ],
            [
                "I can't get enough air. I feel like I'm suffocating even sitting still.",
                "My fingernails look slightly blue. This has been going on for about 20 minutes.",
                "I have COPD and take tiotropium and fluticasone daily.",
            ],
            [
                "My breathing is so bad I can only say a few words at a time.",
                "I have no known lung disease but I suddenly cannot breathe properly.",
                "This started about 15 minutes ago. No chest pain, just severe shortness of breath.",
            ],
            [
                "I'm gasping for air. I had a cold for a few days and now I cannot breathe.",
                "My oxygen levels on my home monitor say 88 percent.",
                "I feel very weak and my lips are turning blue.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    # ── bleeding ─────────────────────────────────────────────────────────────
    {
        "category": "acute_bleeding",
        "red_flag_category": "bleeding",
        "description": "Patient with significant active bleeding",
        "turn_sets": [
            [
                "I cut my hand badly and the bleeding won't stop. I've been pressing on it for 20 minutes.",
                "It's a deep laceration. The cloth I'm using is completely soaked through.",
                "I'm on blood thinners — warfarin — for a clotting disorder.",
            ],
            [
                "I started coughing up blood about 30 minutes ago. It keeps happening.",
                "There's quite a bit, maybe a couple tablespoons each time I cough.",
                "I'm a smoker, 55 years old. I have never had this happen before.",
            ],
            [
                "I've been noticing blood in my stool for the past day and now I feel really dizzy.",
                "The blood looks dark red. I'm also feeling very weak and lightheaded when I stand.",
                "I take ibuprofen regularly for back pain and have a history of ulcers.",
            ],
            [
                "I'm having extremely heavy vaginal bleeding that soaked through a pad in 30 minutes.",
                "This is much heavier than any period I've ever had. I feel faint.",
                "I'm 34 years old. I had a miscarriage two weeks ago.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    # ── suicidal_ideation ────────────────────────────────────────────────────
    {
        "category": "mental_health_crisis",
        "red_flag_category": "suicidal_ideation",
        "description": "Patient expressing suicidal ideation",
        "turn_sets": [
            [
                "I just can't go on anymore. Everything feels completely pointless.",
                "I've been thinking about ending things. I have a plan.",
                "I don't see any other way out of the pain I'm in.",
            ],
            [
                "I've been having thoughts about ending my life. They've been getting stronger.",
                "I have pills at home and I've been thinking about using them.",
                "I don't want to be here anymore. No one would miss me.",
            ],
            [
                "There's no point in living. I feel like a burden to everyone around me.",
                "I've been researching ways to hurt myself online for the past week.",
                "I haven't slept in days. I feel completely hopeless.",
            ],
            [
                "I gave away my belongings today because I don't need them anymore.",
                "I have been writing goodbye notes. I can't take the pain anymore.",
                "I have struggled with depression for years but it has never been this bad.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "medical_history", "current_medications"],
    },
    # ── anaphylaxis ───────────────────────────────────────────────────────────
    {
        "category": "acute_anaphylaxis",
        "red_flag_category": "anaphylaxis",
        "description": "Patient with severe allergic reaction",
        "turn_sets": [
            [
                "I just ate shrimp and now I have hives all over my body and my throat is tightening.",
                "The hives appeared within minutes. My tongue feels swollen.",
                "I have a known shellfish allergy but I accidentally ate some. I don't have my epipen.",
            ],
            [
                "I took a new antibiotic about 20 minutes ago and my lips are swelling up.",
                "I also have a rash spreading across my chest and my throat feels tight.",
                "I'm having trouble swallowing. I feel very dizzy and my blood pressure might be dropping.",
            ],
            [
                "After eating at a restaurant I suddenly have severe hives and swelling around my eyes.",
                "I'm having trouble breathing. My face is puffing up rapidly.",
                "This has happened before with peanuts. I used my epipen but still feel very bad.",
            ],
            [
                "I was stung by a bee and I've never reacted like this. My throat is closing.",
                "I feel extremely dizzy and I have hives everywhere.",
                "I'm very short of breath and my heart is racing. This happened very fast.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "allergies"],
    },
]


# ---------------------------------------------------------------------------
# Routine (non-red-flag) turn templates — English
# ---------------------------------------------------------------------------

ROUTINE_TEMPLATES: List[Dict[str, Any]] = [
    # ── upper_respiratory ─────────────────────────────────────────────────────
    {
        "category": "upper_respiratory",
        "description": "Patient with cold or upper respiratory symptoms",
        "turn_sets": [
            [
                "I've had a runny nose and sore throat for about three days.",
                "I also have a mild cough and feel a bit tired. No fever.",
                "I take a daily antihistamine for seasonal allergies.",
            ],
            [
                "I started feeling sick two days ago with a sore throat and sneezing.",
                "My nose is very congested and I have a mild headache.",
                "I've had a low-grade fever of about 99 degrees Fahrenheit.",
            ],
            [
                "I have a persistent cough and nasal congestion for about five days.",
                "My throat hurts when I swallow and I feel generally run down.",
                "I'm allergic to penicillin. Otherwise I don't take regular medications.",
            ],
            [
                "I woke up this morning with a really bad sore throat and congestion.",
                "I feel achy all over and have a mild fever.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "current_medications", "allergies"],
    },
    # ── gastrointestinal ──────────────────────────────────────────────────────
    {
        "category": "gastrointestinal",
        "description": "Patient with stomach or bowel complaints",
        "turn_sets": [
            [
                "I've had stomach cramps and diarrhea since yesterday evening.",
                "I think it might be something I ate. I've been to the bathroom maybe six times today.",
                "I feel nauseous but haven't vomited. I'm trying to stay hydrated.",
            ],
            [
                "I have abdominal pain, mostly in the lower right area. It started this morning.",
                "The pain is about a 5 out of 10. It gets worse when I press on it.",
                "I've been a bit nauseous and haven't had an appetite.",
            ],
            [
                "I've been constipated for almost a week. Nothing seems to be helping.",
                "I also have bloating and cramping. I feel very uncomfortable.",
                "I'm 45 years old with no significant medical history.",
            ],
            [
                "I have been vomiting since last night, about every two hours.",
                "I can't keep anything down, not even water. I feel very weak.",
                "I have mild cramping in my stomach as well.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # ── musculoskeletal ───────────────────────────────────────────────────────
    {
        "category": "musculoskeletal",
        "description": "Patient with joint or muscle pain",
        "turn_sets": [
            [
                "I twisted my ankle playing basketball yesterday and it's really swollen.",
                "The pain is about a 6 out of 10 when I put weight on it.",
                "I can walk but it really hurts. There's bruising around the outside.",
            ],
            [
                "I've had lower back pain for about two weeks. It started after moving furniture.",
                "The pain is worse in the morning and when I sit for long periods.",
                "I take ibuprofen occasionally which helps a bit.",
            ],
            [
                "My right knee has been aching for several months, getting gradually worse.",
                "I'm 62 and I think it might be arthritis. The stiffness is worst in the morning.",
                "Over-the-counter pain relievers help somewhat.",
            ],
            [
                "I have severe neck pain that started this morning when I woke up.",
                "I can barely turn my head to either side. It feels very stiff.",
                "I think I slept in a bad position.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "current_medications"],
    },
    # ── skin ──────────────────────────────────────────────────────────────────
    {
        "category": "skin",
        "description": "Patient with skin rash or lesion",
        "turn_sets": [
            [
                "I have a red itchy rash on both forearms that appeared about two days ago.",
                "It's spreading slowly upward. I was gardening recently and touched some plants.",
                "I have no known plant allergies but this is new.",
            ],
            [
                "I have a painful blister-like rash on the left side of my torso.",
                "It started as a tingling sensation three days ago and now it really hurts.",
                "I'm 67 and I had chickenpox as a child.",
            ],
            [
                "I have acne that has been getting much worse over the past few months.",
                "I've tried over-the-counter products but they aren't helping.",
                "I'm 19 years old. The breakouts are mostly on my face and back.",
            ],
            [
                "I noticed a mole on my arm that has changed color and shape over the past few months.",
                "It used to be small and brown but now it's larger with irregular edges.",
                "I spend a lot of time outdoors and have a family history of skin cancer.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # ── headache_routine ──────────────────────────────────────────────────────
    {
        "category": "headache_routine",
        "description": "Patient with recurring or tension headache",
        "turn_sets": [
            [
                "I've been getting headaches almost every day for the past two weeks.",
                "They feel like a band squeezing around my head. Usually a 4 out of 10.",
                "I've been very stressed at work and staring at screens all day.",
            ],
            [
                "I have a migraine that started this morning. I get them occasionally.",
                "The pain is on the left side of my head, throbbing, about a 7 out of 10.",
                "I'm sensitive to light and sound. I took sumatriptan but it's only partially helping.",
            ],
            [
                "I've had a headache behind my eyes for three days.",
                "I also have some nasal congestion. I think it might be sinus related.",
                "I get these sometimes in the winter months.",
            ],
            [
                "I get headaches regularly around my menstrual cycle.",
                "This one is particularly bad, throbbing on one side, nausea as well.",
                "I usually take ibuprofen but it's not touching this one.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "associated_symptoms", "current_medications"],
    },
    # ── urinary ───────────────────────────────────────────────────────────────
    {
        "category": "urinary",
        "description": "Patient with urinary symptoms",
        "turn_sets": [
            [
                "It burns when I urinate and I feel like I need to go all the time.",
                "This has been going on for about two days. The urine looks a bit cloudy.",
                "I'm 28 years old female. I've had UTIs before and this feels the same.",
            ],
            [
                "I've had frequent urination and some burning for about three days.",
                "There might be a little blood in my urine, it looks pinkish.",
                "I'm not on any regular medications. No known allergies.",
            ],
            [
                "I have pain in my lower back and side that started yesterday, and fever.",
                "I also have burning when I urinate. The back pain is pretty severe.",
                "I'm worried it might have moved to my kidneys.",
            ],
            [
                "I keep needing to urinate urgently but very little comes out each time.",
                "I'm a 65-year-old male. I have had prostate issues in the past.",
                "This has been gradually worsening over the past few months.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # ── chronic_diabetes ──────────────────────────────────────────────────────
    {
        "category": "chronic_diabetes",
        "description": "Patient managing or newly symptomatic with diabetes",
        "turn_sets": [
            [
                "I'm a type 2 diabetic and my blood sugars have been running really high this week.",
                "My fasting readings are around 250 to 280. I've been more thirsty than usual.",
                "I take metformin 1000 mg twice daily. I may have missed a few doses.",
            ],
            [
                "I have type 1 diabetes and I'm feeling shaky and sweaty right now.",
                "My glucose monitor shows 58 mg/dL. I've had juice but it's not coming up.",
                "I'm on insulin — basal-bolus regimen.",
            ],
            [
                "I've been having unusual fatigue, excessive thirst, and frequent urination for a month.",
                "I've also lost about 10 pounds without trying.",
                "I have a strong family history of type 2 diabetes. I haven't been tested yet.",
            ],
            [
                "My diabetes has been hard to control lately. My A1C was 9.5 at my last visit.",
                "I've been having numbness and tingling in my feet for a few months.",
                "I take metformin and glipizide. I also have hypertension.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    # ── chronic_hypertension ──────────────────────────────────────────────────
    {
        "category": "chronic_hypertension",
        "description": "Patient with blood pressure management concern",
        "turn_sets": [
            [
                "My blood pressure has been very high at home, readings around 170 over 100.",
                "I've had a headache and feel a bit off. This has been for about two days.",
                "I take amlodipine and lisinopril but ran out of lisinopril three days ago.",
            ],
            [
                "I check my blood pressure at home and today it was 185 over 115.",
                "I feel a little lightheaded but no chest pain. I have hypertension diagnosed five years ago.",
                "I'm on hydrochlorothiazide and recently started a new diet.",
            ],
            [
                "I was told I have high blood pressure at a health screening yesterday.",
                "I haven't been on any medication for it. The reading was 160 over 95.",
                "I'm 52, I smoke about half a pack a day, and my father had a stroke.",
            ],
            [
                "My blood pressure medication makes me feel dizzy when I stand up.",
                "I've had these spells for a few weeks since my doctor adjusted my dose.",
                "I'm on metoprolol and amlodipine. I'm 68 years old.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    # ── pediatric_fever ───────────────────────────────────────────────────────
    {
        "category": "pediatric_fever",
        "description": "Parent reporting child with fever",
        "turn_sets": [
            [
                "My daughter has had a fever since last night. It got up to 103.2 degrees Fahrenheit.",
                "She's four years old. She also has a runny nose and isn't eating much.",
                "I gave her children's Tylenol which brought it down a bit but it keeps coming back.",
            ],
            [
                "My son is seven and has had a fever for two days, around 101 to 102.",
                "He also has a sore throat and says it hurts to swallow.",
                "He doesn't have any known allergies and isn't on any medication.",
            ],
            [
                "My baby is 18 months old and has had a fever of 102 degrees for one day.",
                "She's been fussier than usual and pulling at her right ear.",
                "She has had ear infections before.",
            ],
            [
                "My son is 10 years old and has a fever of 101 and a rash on his trunk.",
                "The rash appeared this morning after he had a fever for one day.",
                "He has no known drug allergies.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "associated_symptoms", "current_medications"],
    },
    # ── medication_sideeffect ─────────────────────────────────────────────────
    {
        "category": "medication_sideeffect",
        "description": "Patient reporting medication side effect",
        "turn_sets": [
            [
                "I started a new antibiotic three days ago and my stomach has been really upset.",
                "I have nausea, cramping, and diarrhea since starting it.",
                "The antibiotic is amoxicillin-clavulanate 875 mg twice daily.",
            ],
            [
                "I started a statin two weeks ago and I've been having muscle aches all over.",
                "The muscle pain is mild but persistent. I'm a bit worried.",
                "The medication is atorvastatin 40 mg. I have high cholesterol and was just started on it.",
            ],
            [
                "My new blood pressure medication is giving me a persistent dry cough.",
                "It started about a week after beginning lisinopril. It's annoying and keeps me up at night.",
                "I have hypertension and no other significant medical history.",
            ],
            [
                "I think I'm having a reaction to my new antidepressant.",
                "I feel restless and can't sit still. My legs feel like they need to keep moving.",
                "The medication is sertraline 50 mg, started about 10 days ago.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "current_medications"],
    },
    # ── mental_health_routine ─────────────────────────────────────────────────
    {
        "category": "mental_health_routine",
        "description": "Patient with fatigue or low mood without acute crisis",
        "turn_sets": [
            [
                "I've been feeling really exhausted and down lately, for the past few weeks.",
                "I'm not sleeping well, waking up at 3 in the morning and can't get back to sleep.",
                "I've lost interest in things I usually enjoy. No thoughts of harming myself.",
            ],
            [
                "I've been feeling anxious almost every day for several months.",
                "My heart races and I get sweaty in social situations. I avoid going out.",
                "I've never been treated for anxiety. I don't take any medications.",
            ],
            [
                "I feel overwhelmed and stressed all the time. Work has been very demanding.",
                "I've had headaches and trouble concentrating. My appetite has changed.",
                "I sleep too much on weekends trying to catch up.",
            ],
            [
                "I've been feeling very low and unmotivated for about a month.",
                "I cry frequently without a clear reason. My energy is very low.",
                "I'm 35 years old and I had postpartum depression after my last child.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history"],
    },
    # ── prenatal_routine ──────────────────────────────────────────────────────
    {
        "category": "prenatal_routine",
        "description": "Pregnant patient with routine prenatal concern",
        "turn_sets": [
            [
                "I'm 28 weeks pregnant and I've been having swelling in my ankles and feet.",
                "It started about a week ago. My blood pressure at home has been normal.",
                "I also feel very tired and have mild lower back aching.",
            ],
            [
                "I'm 10 weeks pregnant and the morning sickness is really severe.",
                "I've been vomiting multiple times a day and can barely keep food down.",
                "I haven't gained any weight and feel quite weak.",
            ],
            [
                "I'm 34 weeks pregnant and I've noticed my baby hasn't been moving as much today.",
                "Usually I feel kicks throughout the day but today it's been very quiet.",
                "I'm otherwise feeling okay, no pain or bleeding.",
            ],
            [
                "I'm 20 weeks pregnant and I have a burning sensation in my chest after eating.",
                "I know heartburn is common in pregnancy but it's quite uncomfortable.",
                "I'm also having some round ligament pain on the right side.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history"],
    },
]


# ---------------------------------------------------------------------------
# Spanish translations for red-flag categories
# ---------------------------------------------------------------------------

RED_FLAG_TEMPLATES_ES: List[Dict[str, Any]] = [
    {
        "category": "acute_cardiac",
        "red_flag_category": "cardiac",
        "description": "Adulto con malestar toracico agudo",
        "turn_sets": [
            [
                "Tengo una presion muy fuerte en el pecho desde hace una hora.",
                "El dolor es como un 7 de 10. Me esta doliendo el brazo izquierdo tambien.",
                "Tuve un infarto hace dos anos y tomo aspirina y metoprolol.",
            ],
            [
                "Siento que alguien me aprieta el pecho. Estoy sudando mucho sin razon.",
                "El dolor sube hacia mi mandibula. Es muy severo, como un 9 de 10.",
            ],
            [
                "Tengo palpitaciones y luego me mareo mucho. Me ha pasado tres veces hoy.",
                "Mi corazon late muy rapido e irregular. Tengo mucho miedo.",
                "Tengo hipertension y tomo amlodipino.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "medical_history", "current_medications"],
    },
    {
        "category": "acute_stroke",
        "red_flag_category": "stroke",
        "description": "Paciente con sintomas neurologicos repentinos",
        "turn_sets": [
            [
                "De repente mi brazo derecho se debilito mucho y tengo dificultad para hablar.",
                "Empezo hace 30 minutos. Mi esposa dice que mi cara esta caida de un lado.",
                "Tengo 65 anos y tomo warfarina por fibrilacion auricular.",
            ],
            [
                "Tengo el peor dolor de cabeza de mi vida y la mitad de mi cara esta entumecida.",
                "Empezo de repente hace unos 20 minutos. No he tenido esto antes.",
            ],
            [
                "De repente no puedo entender lo que me dicen. Las palabras no me salen bien.",
                "Mi pierna derecha tambien esta muy debil. Tengo 72 anos.",
                "Esto empezo hace como 15 minutos, completamente de la nada.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "medical_history", "current_medications"],
    },
    {
        "category": "acute_respiratory_distress",
        "red_flag_category": "respiratory_distress",
        "description": "Paciente con dificultad respiratoria severa",
        "turn_sets": [
            [
                "Casi no puedo respirar. He usado mi inhalador cuatro veces y no ayuda.",
                "Mis labios se estan poniendo azules. No puedo terminar una oracion sin parar.",
                "Tengo asma desde nino. Nunca me ha pasado algo tan malo.",
            ],
            [
                "Me estoy ahogando. Llevo 20 minutos sin poder respirar bien.",
                "Tengo EPOC y tomo broncodilatadores pero no estan funcionando ahora.",
            ],
            [
                "Solo puedo decir unas pocas palabras antes de quedarme sin aire.",
                "No tengo enfermedad pulmonar conocida pero de repente no puedo respirar.",
                "Esto empezo hace 15 minutos. Sin dolor en el pecho, solo falta de aire grave.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    {
        "category": "acute_bleeding",
        "red_flag_category": "bleeding",
        "description": "Paciente con sangrado activo significativo",
        "turn_sets": [
            [
                "Estoy tosiendo sangre desde hace media hora. Es bastante cantidad.",
                "Nunca me habia pasado esto. Soy fumador de 55 anos.",
                "Me siento muy debil y asustado.",
            ],
            [
                "Tengo sangre en las heces y me siento muy mareado.",
                "Tomo ibuprofeno regularmente para el dolor de espalda y tengo ulceras.",
                "El mareo empeora cuando me levanto.",
            ],
            [
                "Me corte la mano profundamente y el sangrado no para. Llevo 20 minutos apretando.",
                "El trapo que uso esta completamente empapado de sangre.",
                "Tomo warfarina por un trastorno de coagulacion.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    {
        "category": "mental_health_crisis",
        "red_flag_category": "suicidal_ideation",
        "description": "Paciente con ideacion suicida",
        "turn_sets": [
            [
                "Ya no puedo mas. Todo se siente sin sentido y sin esperanza.",
                "He estado pensando en quitarme la vida. Tengo un plan.",
                "Llevo semanas sintiendome asi, cada dia peor.",
            ],
            [
                "No hay razon para seguir viviendo. Soy una carga para todos.",
                "He estado buscando formas de hacerme dano en internet.",
                "Tengo depresion desde hace anos pero nunca fue tan grave.",
            ],
            [
                "Reparti mis pertenencias hoy porque ya no las necesito.",
                "He estado escribiendo cartas de despedida. No aguanto mas el dolor.",
                "Llevo dias sin dormir. Me siento completamente sin esperanza.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "medical_history", "current_medications"],
    },
    {
        "category": "acute_anaphylaxis",
        "red_flag_category": "anaphylaxis",
        "description": "Paciente con reaccion alergica severa",
        "turn_sets": [
            [
                "Comi mariscos y ahora tengo ronchas por todo el cuerpo y la garganta se me cierra.",
                "Mi lengua esta hinchada. Me esta costando tragar.",
                "Soy alergico a los mariscos pero los comi por accidente. No tengo epinefrina.",
            ],
            [
                "Tome un antibiotico nuevo y mis labios se estan hinchando mucho.",
                "Tambien tengo sarpullido en el pecho y dificultad para respirar.",
                "Me siento muy mareado y mi corazon esta acelerado.",
            ],
            [
                "Me pico una abeja y nunca habia reaccionado asi. Mi garganta se esta cerrando.",
                "Me siento extremadamente mareado y tengo ronchas por todas partes.",
                "Me falta el aire y el corazon me late muy rapido.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "allergies"],
    },
]


# ---------------------------------------------------------------------------
# Spanish translations for routine categories
# ---------------------------------------------------------------------------

ROUTINE_TEMPLATES_ES: List[Dict[str, Any]] = [
    {
        "category": "upper_respiratory",
        "description": "Paciente con sintomas respiratorios altos",
        "turn_sets": [
            [
                "Llevo tres dias con la nariz mocosa y la garganta irritada.",
                "Tambien tengo tos leve y me siento cansado. Sin fiebre.",
                "Tomo antihistaminico para alergias estacionales.",
            ],
            [
                "Me desperte esta manana con mucha congestion nasal y dolor de garganta.",
                "Tengo fiebre leve de 37.8 grados centigrados.",
            ],
            [
                "Tengo tos persistente y congestion nasal desde hace cinco dias.",
                "Me duele la garganta al tragar y me siento muy cansado.",
                "Soy alergico a la penicilina. No tomo otros medicamentos habitualmente.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "current_medications"],
    },
    {
        "category": "gastrointestinal",
        "description": "Paciente con molestias gastrointestinales",
        "turn_sets": [
            [
                "Llevo un dia con colicos estomacales y diarrea. Creo que fue algo que comi.",
                "He ido al bano unas seis veces hoy. Me siento debil.",
                "Estoy intentando tomar liquidos para no deshidratarme.",
            ],
            [
                "Tengo dolor abdominal en el lado inferior derecho desde esta manana.",
                "El dolor empeora cuando presiono la zona. Tambien tengo nauseas.",
            ],
            [
                "Llevo vomitando desde anoche, cada dos horas aproximadamente.",
                "No puedo retener nada, ni siquiera agua. Me siento muy debil.",
                "Tambien tengo calambres leves en el estomago.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    {
        "category": "pediatric_fever",
        "description": "Padre reportando fiebre en hijo",
        "turn_sets": [
            [
                "Mi hija tiene fiebre desde anoche, llego a 39.5 grados centigrados.",
                "Tiene cuatro anos. Tambien tiene mocos y no quiere comer.",
                "Le di paracetamol infantil y bajo un poco pero vuelve a subir.",
            ],
            [
                "Mi hijo de siete anos lleva dos dias con fiebre entre 38 y 39 grados.",
                "Tambien le duele la garganta y dice que le cuesta tragar.",
                "No tiene alergias conocidas y no toma medicamentos.",
            ],
            [
                "Mi bebe tiene 18 meses y lleva un dia con fiebre de 39 grados.",
                "Esta mas irritable de lo normal y se jala la oreja derecha.",
                "Ha tenido infecciones de oido antes.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "associated_symptoms", "current_medications"],
    },
    {
        "category": "musculoskeletal",
        "description": "Paciente con dolor muscular o articular",
        "turn_sets": [
            [
                "Me torci el tobillo jugando futbol ayer y esta muy hinchado.",
                "El dolor es como un 6 de 10 cuando apoyo el pie.",
                "Hay moretones alrededor del tobillo externo.",
            ],
            [
                "Llevo dos semanas con dolor lumbar que empezo al mover muebles.",
                "Es peor por las mananas y cuando estoy sentado mucho tiempo.",
                "Tomo ibuprofeno ocasionalmente.",
            ],
            [
                "Mi rodilla derecha ha estado doliendo por varios meses, empeorando gradualmente.",
                "Tengo 62 anos y creo que puede ser artritis. La rigidez es peor por las mananas.",
                "Los analgesicos de venta libre ayudan un poco.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "current_medications"],
    },
    {
        "category": "chronic_diabetes",
        "description": "Paciente con diabetes descompensada",
        "turn_sets": [
            [
                "Soy diabetico tipo 2 y mis glucosas han estado muy altas esta semana.",
                "En ayunas tengo entre 250 y 280. Tengo mas sed de lo normal.",
                "Tomo metformina 1000 mg dos veces al dia. Quiza me salte algunas dosis.",
            ],
            [
                "Tengo diabetes tipo 1 y ahora mismo me siento tembloroso y sudoroso.",
                "Mi glucometro marca 58. Tome jugo pero no sube.",
                "Estoy en insulina, regimen basal-bolo.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    {
        "category": "mental_health_routine",
        "description": "Paciente con fatiga o estado animo bajo sin crisis",
        "turn_sets": [
            [
                "Me he sentido muy agotado y triste ultimamente, durante las ultimas semanas.",
                "No duermo bien, me despierto a las 3 de la manana y no puedo volver a dormir.",
                "He perdido el interes en cosas que normalmente disfruto. Sin pensamientos de dano.",
            ],
            [
                "Me he sentido ansioso casi todos los dias durante varios meses.",
                "El corazon se me acelera y sudo en situaciones sociales. Evito salir.",
                "Nunca he sido tratado por ansiedad. No tomo medicamentos.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history"],
    },
]


# ---------------------------------------------------------------------------
# French translations for red-flag categories
# ---------------------------------------------------------------------------

RED_FLAG_TEMPLATES_FR: List[Dict[str, Any]] = [
    {
        "category": "acute_cardiac",
        "red_flag_category": "cardiac",
        "description": "Adulte avec douleur thoracique aigue",
        "turn_sets": [
            [
                "J'ai une forte pression dans la poitrine depuis environ une heure.",
                "La douleur est de 7 sur 10. Mon bras gauche est egalement engourdi.",
                "J'ai eu une crise cardiaque il y a deux ans et je prends de l'aspirine.",
            ],
            [
                "J'ai une douleur qui remonte dans la machoire et je transpire beaucoup.",
                "C'est tres severe, environ 9 sur 10. J'ai tres peur.",
            ],
            [
                "J'ai des palpitations puis je me sens etourdi. C'est arrive trois fois aujourd'hui.",
                "Mon coeur bat tres vite puis de facon irreguliere.",
                "J'ai de l'hypertension et je prends de l'amlodipine.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "medical_history", "current_medications"],
    },
    {
        "category": "acute_stroke",
        "red_flag_category": "stroke",
        "description": "Patient avec symptomes neurologiques soudains",
        "turn_sets": [
            [
                "Mon bras droit est soudainement devenu tres faible il y a 30 minutes.",
                "Ma parole est brouillee aussi. Ma femme dit que mon visage est tombant d'un cote.",
                "J'ai 68 ans et je prends de la warfarine pour une fibrillation auriculaire.",
            ],
            [
                "J'ai le pire mal de tete de ma vie et la moitie de mon visage est engourdie.",
                "Cela a commence soudainement il y a 20 minutes.",
            ],
            [
                "Je me suis reveille et soudainement je ne comprends plus ce qu'on me dit.",
                "J'essaie de parler mais les mots ne sortent pas correctement.",
                "Ma jambe droite est aussi tres faible. J'ai 72 ans.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "medical_history", "current_medications"],
    },
    {
        "category": "acute_respiratory_distress",
        "red_flag_category": "respiratory_distress",
        "description": "Patient avec difficultes respiratoires severes",
        "turn_sets": [
            [
                "Je peux a peine respirer. Mon inhalateur ne m'aide plus du tout.",
                "Mes levres deviennent bleues. Je ne peux pas finir une phrase sans m'arreter.",
                "J'ai de l'asthme depuis l'enfance. C'est la pire crise que j'ai eue.",
            ],
            [
                "Je suffoque. Cela dure depuis 20 minutes et ca s'aggrave.",
                "J'ai une BPCO et mes bronchodilatateurs ne fonctionnent pas maintenant.",
            ],
            [
                "Je ne peux dire que quelques mots avant de m'essouffler.",
                "Je n'ai pas de maladie pulmonaire connue mais je ne peux soudainement plus respirer.",
                "Cela a commence il y a 15 minutes. Pas de douleur thoracique, juste un essoufflement severe.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    {
        "category": "acute_bleeding",
        "red_flag_category": "bleeding",
        "description": "Patient avec saignement actif significatif",
        "turn_sets": [
            [
                "Je crache du sang depuis une demi-heure. C'est assez abondant.",
                "Cela ne m'est jamais arrive. J'ai 55 ans et je fume.",
                "Je me sens tres faible et apeure.",
            ],
            [
                "J'ai du sang dans les selles et je me sens tres etourdi.",
                "Je prends de l'ibuprofene regulierement et j'ai des antecedents d'ulceres.",
            ],
            [
                "Je me suis profondement coupe la main et le saignement ne s'arrete pas.",
                "J'ai appuye dessus pendant 20 minutes mais le tissu est completement trempe.",
                "Je prends de la warfarine pour un trouble de coagulation.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    {
        "category": "mental_health_crisis",
        "red_flag_category": "suicidal_ideation",
        "description": "Patient exprimant des idees suicidaires",
        "turn_sets": [
            [
                "Je n'en peux plus. Tout me semble inutile et sans espoir.",
                "J'ai pense a en finir avec ma vie. J'ai un plan.",
                "Je me sens comme ca depuis des semaines, chaque jour c'est pire.",
            ],
            [
                "Il n'y a plus aucune raison de vivre. Je suis un fardeau pour tout le monde.",
                "J'ai cherche des moyens de me faire du mal sur internet.",
            ],
            [
                "J'ai distribue mes affaires aujourd'hui car je n'en aurai plus besoin.",
                "J'ai ecrit des lettres d'adieu. Je ne supporte plus la douleur.",
                "J'ai lutte contre la depression pendant des annees mais jamais aussi grave.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "medical_history", "current_medications"],
    },
    {
        "category": "acute_anaphylaxis",
        "red_flag_category": "anaphylaxis",
        "description": "Patient avec reaction allergique severe",
        "turn_sets": [
            [
                "J'ai mange des crevettes et j'ai maintenant des urticaires partout et ma gorge se resserre.",
                "Ma langue est gonflee. J'ai du mal a avaler.",
                "Je suis allergique aux crustaces mais j'en ai mange par accident.",
            ],
            [
                "J'ai pris un nouvel antibiotique et mes levres gonflent rapidement.",
                "J'ai aussi une eruption cutanee sur la poitrine et du mal a respirer.",
                "Je me sens tres etourdi et mon coeur bat tres vite.",
            ],
            [
                "J'ai ete pique par une abeille et ma gorge se ferme. Je n'ai jamais reagi ainsi.",
                "Je me sens extremement etourdi et j'ai des urticaires partout.",
                "J'ai du mal a respirer et mon coeur s'emballe. C'est arrive tres vite.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "allergies"],
    },
]


# ---------------------------------------------------------------------------
# French translations for routine categories
# ---------------------------------------------------------------------------

ROUTINE_TEMPLATES_FR: List[Dict[str, Any]] = [
    {
        "category": "upper_respiratory",
        "description": "Patient avec symptomes respiratoires hauts",
        "turn_sets": [
            [
                "J'ai le nez qui coule et la gorge irritee depuis trois jours.",
                "J'ai aussi une legere toux et je me sens fatigue. Pas de fievre.",
                "Je prends un antihistaminique pour les allergies saisonnieres.",
            ],
            [
                "Je me suis reveille ce matin avec une forte congestion nasale et mal a la gorge.",
                "J'ai une legere fievre de 38 degres.",
            ],
            [
                "J'ai une toux persistante et une congestion nasale depuis environ cinq jours.",
                "J'ai mal a la gorge quand j'avale et je me sens generalement fatigue.",
                "Je suis allergique a la penicilline. Sinon je ne prends pas de medicaments reguliers.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "current_medications"],
    },
    {
        "category": "gastrointestinal",
        "description": "Patient avec plaintes gastrointestinales",
        "turn_sets": [
            [
                "J'ai des crampes abdominales et de la diarrhee depuis hier soir.",
                "Je suis alle aux toilettes environ six fois aujourd'hui. Je me sens faible.",
                "J'essaie de boire beaucoup pour ne pas me deshydrater.",
            ],
            [
                "J'ai une douleur abdominale en bas a droite depuis ce matin.",
                "La douleur s'aggrave quand j'appuie dessus. J'ai aussi des nausees.",
            ],
            [
                "Je vomis depuis hier soir, environ toutes les deux heures.",
                "Je ne peux rien garder, pas meme de l'eau. Je me sens tres faible.",
                "J'ai aussi de legeres crampes dans l'estomac.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    {
        "category": "pediatric_fever",
        "description": "Parent signalant de la fievre chez l'enfant",
        "turn_sets": [
            [
                "Ma fille a de la fievre depuis hier soir, elle a atteint 39,5 degres.",
                "Elle a quatre ans. Elle a aussi le nez qui coule et ne mange pas beaucoup.",
                "Je lui ai donne du paracetamol pour enfants et elle a un peu baisse.",
            ],
            [
                "Mon fils de sept ans a de la fievre depuis deux jours, entre 38 et 39 degres.",
                "Il a aussi mal a la gorge et dit que ca fait mal d'avaler.",
            ],
            [
                "Mon bebe a 18 mois et a de la fievre a 39 degres depuis un jour.",
                "Il est plus irritable que d'habitude et se tire l'oreille droite.",
                "Il a deja eu des otites.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "associated_symptoms", "current_medications"],
    },
    {
        "category": "musculoskeletal",
        "description": "Patient avec douleur musculaire ou articulaire",
        "turn_sets": [
            [
                "Je me suis tordu la cheville en jouant au football hier et elle est tres enflee.",
                "La douleur est de 6 sur 10 quand j'appuie dessus.",
                "Il y a des ecchymoses autour de la cheville exterieure.",
            ],
            [
                "J'ai des douleurs lombaires depuis deux semaines qui ont commence en demenageant des meubles.",
                "C'est pire le matin et quand je suis assis longtemps.",
            ],
            [
                "Mon genou droit me fait mal depuis plusieurs mois, en empirant progressivement.",
                "J'ai 62 ans et je pense que c'est peut-etre de l'arthrite. La raideur est pire le matin.",
                "Les analgesiques en vente libre aident un peu.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "current_medications"],
    },
    {
        "category": "chronic_hypertension",
        "description": "Patient avec pression arterielle elevee",
        "turn_sets": [
            [
                "Ma tension arterielle est tres elevee a la maison, autour de 170 sur 100.",
                "J'ai un mal de tete et je ne me sens pas bien. Cela dure depuis deux jours.",
                "Je prends de l'amlodipine et du lisinopril mais je n'ai plus de lisinopril depuis trois jours.",
            ],
            [
                "J'ai verifie ma tension et aujourd'hui c'etait 185 sur 115.",
                "Je me sens un peu etourdi mais pas de douleur thoracique.",
                "J'ai de l'hypertension diagnostiquee il y a cinq ans.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    {
        "category": "mental_health_routine",
        "description": "Patient avec fatigue ou humeur basse sans crise aigue",
        "turn_sets": [
            [
                "Je me sens vraiment epuise et deprime ces derniers temps, depuis quelques semaines.",
                "Je ne dors pas bien, je me reveille a 3 heures du matin et ne peux pas me rendormir.",
                "J'ai perdu interet pour les choses que j'aime habituellement. Pas de pensees de me blesser.",
            ],
            [
                "Je me sens anxieux presque tous les jours depuis plusieurs mois.",
                "Mon coeur s'emballe et je transpire dans les situations sociales. J'evite de sortir.",
                "Je n'ai jamais ete traite pour l'anxiete. Je ne prends pas de medicaments.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history"],
    },
    {
        "category": "prenatal_routine",
        "description": "Patiente enceinte avec probleme prenatal courant",
        "turn_sets": [
            [
                "Je suis a 28 semaines de grossesse et j'ai des gonflements aux chevilles et aux pieds.",
                "Cela a commence il y a environ une semaine. Ma tension arterielle a la maison est normale.",
                "Je me sens aussi tres fatiguee et j'ai de legeres douleurs lombaires.",
            ],
            [
                "Je suis a 10 semaines de grossesse et les nausees matinales sont vraiment severes.",
                "Je vomis plusieurs fois par jour et je peux a peine garder de la nourriture.",
                "Je n'ai pas pris de poids et je me sens tres faible.",
            ],
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history"],
    },
]


# ---------------------------------------------------------------------------
# Edge-case templates (language-tagged, pre-built turns)
# ---------------------------------------------------------------------------

EDGE_CASE_TEMPLATES: List[Dict[str, Any]] = [
    # Minimal input — English
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal single-word input",
        "language": "en",
        "turns": ["Pain."],
        "base_fields": ["chief_complaint"],
    },
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal distress input",
        "language": "en",
        "turns": ["Help."],
        "base_fields": ["chief_complaint"],
    },
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal two-word input",
        "language": "en",
        "turns": ["Chest hurts."],
        "base_fields": ["chief_complaint"],
    },
    # Minimal input — Spanish
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal Spanish input",
        "language": "es",
        "turns": ["Dolor."],
        "base_fields": ["chief_complaint"],
    },
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal Spanish distress",
        "language": "es",
        "turns": ["Ayuda."],
        "base_fields": ["chief_complaint"],
    },
    # Minimal input — French
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal French input",
        "language": "fr",
        "turns": ["Aide."],
        "base_fields": ["chief_complaint"],
    },
    {
        "category": "unknown_minimal",
        "red_flag_category": None,
        "description": "Minimal French pain input",
        "language": "fr",
        "turns": ["Douleur."],
        "base_fields": ["chief_complaint"],
    },
    # Very long description — upper respiratory
    {
        "category": "upper_respiratory",
        "red_flag_category": None,
        "description": "Very long patient description over 150 words",
        "language": "en",
        "turns": [
            (
                "I've been feeling really unwell for about the past four or five days now and I wasn't sure "
                "whether I should call or come in but my symptoms just don't seem to be improving. It started "
                "with a scratchy throat on Monday morning and by Tuesday I had a full-blown runny nose, "
                "sneezing constantly, and a mild headache mostly behind my eyes and forehead. My body has "
                "been achy, especially my muscles in my legs and lower back, and I've been fatigued to the "
                "point where I'm going to bed by 8 PM every night which is very unlike me. I've been taking "
                "ibuprofen 400 mg every six hours which helps a little with the body aches and headache but "
                "the congestion is still really bad. I'm also taking a zinc supplement and vitamin C and "
                "drinking a lot of fluids. I don't have a fever that I know of but I feel hot and flushed "
                "sometimes. I have no chest pain and I can breathe okay, just very congested."
            )
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "current_medications"],
    },
    # Very long description — musculoskeletal
    {
        "category": "musculoskeletal",
        "red_flag_category": None,
        "description": "Very long description of back pain history",
        "language": "en",
        "turns": [
            (
                "I've had back pain on and off for years but this current episode has been going on for about "
                "three weeks and it's really affecting my quality of life. The pain is in my lower back, "
                "centered around the L4 L5 area according to my last MRI from about two years ago which showed "
                "a disc bulge. This time the pain is worse than usual and radiates down my right leg all the "
                "way to my foot which I don't normally get. The tingling in my foot is new and concerning me. "
                "I work as a nurse so I'm on my feet all day which makes it worse. I've tried ice and heat, "
                "ibuprofen 600 mg three times daily, and a muscle relaxant that my doctor prescribed last time. "
                "Nothing is helping as much as usual. I'm 44 years old, female, non-smoker. I'm also on "
                "levothyroxine for hypothyroidism. I'm wondering if I need another MRI or if there's something "
                "else going on that's different from my usual disc problem."
            )
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity", "associated_symptoms", "medical_history", "current_medications"],
    },
    # Very long description — chronic diabetes
    {
        "category": "chronic_diabetes",
        "red_flag_category": None,
        "description": "Very long diabetes management description",
        "language": "en",
        "turns": [
            (
                "I've been a type 2 diabetic for about 12 years and lately my blood sugar control has been "
                "really poor. My endocrinologist adjusted my medications about two months ago — I was on "
                "metformin only and she added sitagliptin and also started me on a low dose of insulin "
                "glargine at night. Since then my morning fasting sugars have been okay, usually around 110 "
                "to 130, but my after-meal readings are still running very high, sometimes 240 to 280. I've "
                "been trying to follow a low-carbohydrate diet but it's difficult. I'm also having numbness "
                "and tingling in both feet that has been getting gradually worse over the past six months. "
                "My nephrologist says my kidney function is at stage 2 chronic kidney disease. I'm on "
                "lisinopril for blood pressure and kidney protection. I also take atorvastatin and aspirin. "
                "I'm 58 years old and I work a sedentary desk job which I know doesn't help. My last A1C "
                "six weeks ago was 8.2 percent which is lower than before but still not at goal."
            )
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms", "medical_history", "current_medications"],
    },
    # Parent speaking for child — fever and rash
    {
        "category": "pediatric_fever",
        "red_flag_category": None,
        "description": "Parent speaking for young child with fever and rash",
        "language": "en",
        "turns": [
            "I'm calling about my three-year-old son. He started with a fever two days ago, around 38.5 degrees.",
            "This morning he developed a rash on his tummy and back. It's small red spots.",
            "He's been drinking okay but not eating much. He seems tired and a bit clingy.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # Parent speaking for infant
    {
        "category": "pediatric_fever",
        "red_flag_category": None,
        "description": "Parent speaking for infant with feeding difficulty",
        "language": "en",
        "turns": [
            "I'm worried about my 6-month-old baby. She's been very fussy for the past day.",
            "She's not feeding as well as usual and feels warm. I took her temperature and it's 38.2.",
            "She's only having wet diapers every 8 hours instead of her usual every 4.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # Parent speaking for child — Spanish
    {
        "category": "pediatric_fever",
        "red_flag_category": None,
        "description": "Madre hablando por su hijo con fiebre",
        "language": "es",
        "turns": [
            "Llamo por mi hijo de tres anos. Tiene fiebre desde hace dos dias, alrededor de 38.5 grados.",
            "Esta manana le salio un sarpullido en la barriga y la espalda. Son puntitos rojos.",
            "Esta tomando liquidos pero no quiere comer. Se ve cansado y muy pegado a mi.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # Parent speaking for child — French
    {
        "category": "pediatric_fever",
        "red_flag_category": None,
        "description": "Parent parlant pour son enfant avec fievre",
        "language": "fr",
        "turns": [
            "J'appelle pour mon fils de trois ans. Il a de la fievre depuis deux jours, environ 38,5 degres.",
            "Ce matin il a developpe une eruption sur le ventre et le dos. Ce sont de petits points rouges.",
            "Il boit bien mais ne mange pas beaucoup. Il semble fatigue et tres colle a moi.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # Mixed-language — English-Spanish
    {
        "category": "upper_respiratory",
        "red_flag_category": None,
        "description": "Mixed English-Spanish sentence",
        "language": "es",
        "turns": [
            "I have been sick, tengo fiebre y mucho dolor de cabeza since yesterday.",
            "Mi garganta tambien duele a lot when I swallow.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # Mixed-language — English-French
    {
        "category": "gastrointestinal",
        "red_flag_category": None,
        "description": "Mixed English-French sentence",
        "language": "fr",
        "turns": [
            "J'ai mal au ventre, it hurts a lot, depuis ce matin.",
            "J'ai aussi de la nausee and I vomited once.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "associated_symptoms"],
    },
    # Mixed-language — Spanish-French
    {
        "category": "musculoskeletal",
        "red_flag_category": None,
        "description": "Mixed Spanish-French sentence",
        "language": "es",
        "turns": [
            "Tengo dolor en el brazo, ca fait vraiment mal, desde ayer.",
            "No puedo levantar le bras. Es como un 6 de 10.",
        ],
        "base_fields": ["chief_complaint", "symptom_duration", "symptom_severity"],
    },
    # Terse red-flag — cardiac
    {
        "category": "acute_cardiac",
        "red_flag_category": "cardiac",
        "description": "Terse cardiac complaint",
        "language": "en",
        "turns": ["Chest pain. Bad. Started an hour ago."],
        "base_fields": ["chief_complaint", "symptom_duration"],
    },
    # Terse red-flag — suicidal ideation
    {
        "category": "mental_health_crisis",
        "red_flag_category": "suicidal_ideation",
        "description": "Terse suicidal ideation input",
        "language": "en",
        "turns": ["I want to die."],
        "base_fields": ["chief_complaint"],
    },
    # Terse red-flag — stroke
    {
        "category": "acute_stroke",
        "red_flag_category": "stroke",
        "description": "Terse stroke complaint",
        "language": "en",
        "turns": ["Can't move arm. Face drooping. Just started."],
        "base_fields": ["chief_complaint", "symptom_duration"],
    },
    # Terse red-flag — respiratory
    {
        "category": "acute_respiratory_distress",
        "red_flag_category": "respiratory_distress",
        "description": "Terse respiratory distress complaint",
        "language": "en",
        "turns": ["Can't breathe. Lips turning blue."],
        "base_fields": ["chief_complaint", "associated_symptoms"],
    },
    # Terse red-flag — anaphylaxis Spanish
    {
        "category": "acute_anaphylaxis",
        "red_flag_category": "anaphylaxis",
        "description": "Terse anaphylaxis complaint in Spanish",
        "language": "es",
        "turns": ["Reaccion alergica. Garganta cerrandose. Comi cacahuetes."],
        "base_fields": ["chief_complaint", "allergies"],
    },
    # Terse red-flag — bleeding French
    {
        "category": "acute_bleeding",
        "red_flag_category": "bleeding",
        "description": "Terse bleeding complaint in French",
        "language": "fr",
        "turns": ["Je saigne beaucoup. Ca ne s'arrete pas."],
        "base_fields": ["chief_complaint"],
    },
]


# ---------------------------------------------------------------------------
# Demographic helpers
# ---------------------------------------------------------------------------

PEDIATRIC_CATEGORIES = {"pediatric_fever"}
ELDERLY_SKEW_CATEGORIES = {"acute_cardiac", "acute_stroke", "chronic_hypertension", "chronic_diabetes"}
PRENATAL_CATEGORIES = {"prenatal_routine"}
GENDERS = ["male", "female"]


def _pick_demographic(rng: random.Random, category: str) -> Dict[str, Any]:
    """Return a demographic dict with age (int) and gender (str)."""
    gender = rng.choice(GENDERS)
    if category in PRENATAL_CATEGORIES:
        age = rng.randint(18, 44)
        gender = "female"
    elif category in PEDIATRIC_CATEGORIES:
        age = rng.randint(2, 14)
        gender = rng.choice(GENDERS)
    elif category in ELDERLY_SKEW_CATEGORIES:
        if rng.random() < 0.6:
            age = rng.randint(55, 85)
        else:
            age = rng.randint(18, 70)
    else:
        age = rng.randint(18, 80)
    return {"age": age, "gender": gender}


# ---------------------------------------------------------------------------
# Turn-count selection
# ---------------------------------------------------------------------------

def _pick_turn_count(rng: random.Random, turn_set: List[str]) -> List[str]:
    """Select 2-4 turns from a turn set, respecting set length."""
    max_turns = min(4, len(turn_set))
    min_turns = min(2, len(turn_set))
    if min_turns >= max_turns:
        return list(turn_set[:max_turns])
    n = rng.randint(min_turns, max_turns)
    return list(turn_set[:n])


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _build_from_template(
    rng: random.Random,
    template: Dict[str, Any],
    language: str,
    idx: int,
) -> Dict[str, Any]:
    """Build one scenario dict from a structured template."""
    turn_set = rng.choice(template["turn_sets"])
    turns = _pick_turn_count(rng, turn_set)
    category = template["category"]
    red_flag_cat = template.get("red_flag_category")
    demo = _pick_demographic(rng, category)
    return {
        "scenario_id": f"gen_{idx:04d}",
        "language": language,
        "description": template["description"],
        "category": category,
        "turns": turns,
        "expected_fields": list(template["base_fields"]),
        "expected_red_flag": red_flag_cat is not None,
        "red_flag_category": red_flag_cat,
        "demographic": demo,
        "source": "synthetic",
    }


def _build_from_edge_case(edge: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Build one scenario dict from a pre-built edge-case template."""
    rng_local = random.Random(idx * 999983 + 7)
    category = edge["category"]
    demo = _pick_demographic(rng_local, category)
    red_flag_cat = edge.get("red_flag_category")
    return {
        "scenario_id": f"gen_{idx:04d}",
        "language": edge["language"],
        "description": edge["description"],
        "category": category,
        "turns": list(edge["turns"]),
        "expected_fields": list(edge["base_fields"]),
        "expected_red_flag": red_flag_cat is not None,
        "red_flag_category": red_flag_cat,
        "demographic": demo,
        "source": "synthetic",
    }


# ---------------------------------------------------------------------------
# Language splitter
# ---------------------------------------------------------------------------

def _lang_split(rng: random.Random, total: int) -> List[str]:
    """
    Return a shuffled list of language codes for `total` items.
    Distribution: ~60% EN, ~20% ES, ~20% FR.
    """
    n_en = round(total * 0.60)
    n_es = round(total * 0.20)
    n_fr = total - n_en - n_es
    langs = ["en"] * n_en + ["es"] * n_es + ["fr"] * n_fr
    rng.shuffle(langs)
    return langs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_corpus(n: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a deterministic synthetic corpus of clinical intake scenarios.

    Parameters
    ----------
    n : int
        Number of scenarios to generate (default 500).
    seed : int
        Random seed for full reproducibility (default 42).

    Returns
    -------
    list of dict
        Each dict contains:
            scenario_id      : str   — "gen_NNNN" (zero-padded 4 digits)
            language         : str   — "en" | "es" | "fr"
            description      : str   — human-readable description
            category         : str   — clinical category label
            turns            : list  — 1-4 patient message strings
            expected_fields  : list  — extractable field names
            expected_red_flag: bool  — True if scenario has a red flag
            red_flag_category: str|None — red flag type or None
            demographic      : dict  — {"age": int, "gender": str}
            source           : str   — always "synthetic"
    """
    rng = random.Random(seed)
    scenarios: List[Dict[str, Any]] = []

    # ── Determine target counts ───────────────────────────────────────────────
    n_edge = max(1, round(n * 0.10))           # ~10% edge cases
    n_main = n - n_edge
    n_red_flag = round(n_main * 0.35)          # ~35% of main pool are red-flag
    n_routine = n_main - n_red_flag

    # Language splits
    red_flag_langs = _lang_split(rng, n_red_flag)
    routine_langs = _lang_split(rng, n_routine)

    # ── Red-flag scenarios ────────────────────────────────────────────────────
    for lang in red_flag_langs:
        if lang == "en":
            tpl = rng.choice(RED_FLAG_TEMPLATES)
        elif lang == "es":
            tpl = rng.choice(RED_FLAG_TEMPLATES_ES)
        else:
            tpl = rng.choice(RED_FLAG_TEMPLATES_FR)
        sc = _build_from_template(rng, tpl, lang, len(scenarios) + 1)
        scenarios.append(sc)

    # ── Routine scenarios ─────────────────────────────────────────────────────
    for lang in routine_langs:
        if lang == "en":
            tpl = rng.choice(ROUTINE_TEMPLATES)
        elif lang == "es":
            tpl = rng.choice(ROUTINE_TEMPLATES_ES)
        else:
            tpl = rng.choice(ROUTINE_TEMPLATES_FR)
        sc = _build_from_template(rng, tpl, lang, len(scenarios) + 1)
        scenarios.append(sc)

    # ── Edge-case scenarios ───────────────────────────────────────────────────
    edge_pool = list(EDGE_CASE_TEMPLATES)
    rng.shuffle(edge_pool)
    for i in range(n_edge):
        edge_tpl = edge_pool[i % len(edge_pool)]
        sc = _build_from_edge_case(edge_tpl, len(scenarios) + 1)
        scenarios.append(sc)

    # ── Shuffle all and re-assign sequential IDs ──────────────────────────────
    rng.shuffle(scenarios)
    for idx, sc in enumerate(scenarios, start=1):
        sc["scenario_id"] = f"gen_{idx:04d}"

    # Trim to exactly n (rounding may produce n+1 in rare cases)
    return scenarios[:n]


# ---------------------------------------------------------------------------
# Module-level corpus (importable constant, generated once at import time)
# ---------------------------------------------------------------------------

EXTENDED_CORPUS: List[Dict[str, Any]] = generate_corpus(500)


# ---------------------------------------------------------------------------
# Sanity check — run as script: python -m tests.corpus.generator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    corpus = generate_corpus(500, seed=42)
    total = len(corpus)
    red_flags = sum(1 for s in corpus if s["expected_red_flag"])
    by_lang: Dict[str, int] = {}
    for s in corpus:
        by_lang[s["language"]] = by_lang.get(s["language"], 0) + 1
    by_rf_cat: Dict[str, int] = {}
    for s in corpus:
        c = s["red_flag_category"]
        if c:
            by_rf_cat[c] = by_rf_cat.get(c, 0) + 1
    single_turn = sum(1 for s in corpus if len(s["turns"]) == 1)
    edge_count = sum(1 for s in corpus if s["category"] == "unknown_minimal" or len(s["turns"]) == 1)

    print(f"Total scenarios  : {total}")
    print(f"Red flag count   : {red_flags} ({100 * red_flags / total:.1f}%)")
    print(f"Language split   : {by_lang}")
    print(f"Red-flag cats    : {by_rf_cat}")
    print(f"Single-turn      : {single_turn}")
    print(f"Edge-case proxy  : {edge_count}")
    print()
    print("Sample scenario (first):")
    pprint.pprint(corpus[0])
    print()
    print("Sample red-flag scenario:")
    rf_example = next(s for s in corpus if s["expected_red_flag"])
    pprint.pprint(rf_example)
