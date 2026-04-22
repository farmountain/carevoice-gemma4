"""Unit tests for _apply_extraction() and IntakeRecord — no model required."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import pytest
from gemma_hackathon.intake_assistant import _apply_extraction, IntakeRecord


# ---------------------------------------------------------------------------
# GROUP 1 — chief_complaint (15 tests)
# ---------------------------------------------------------------------------

def test_cc_normal_string():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "headache")
    assert r.chief_complaint == "headache"


def test_cc_empty_string():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "")
    assert r.chief_complaint == ""


def test_cc_numeric_coerced_to_str():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", 42)
    assert r.chief_complaint == "42"


def test_cc_float_coerced_to_str():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", 3.14)
    assert r.chief_complaint == "3.14"


def test_cc_very_long_string():
    r = IntakeRecord()
    long_val = "pain " * 100
    _apply_extraction(r, "chief_complaint", long_val)
    assert r.chief_complaint == long_val


def test_cc_unicode_spanish():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "dolor de cabeza")
    assert r.chief_complaint == "dolor de cabeza"


def test_cc_unicode_accents():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "fi\u00e8vre et douleur")
    assert r.chief_complaint == "fi\u00e8vre et douleur"


def test_cc_whitespace_only():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "   ")
    assert r.chief_complaint == "   "


def test_cc_overwrite_existing():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "first value")
    _apply_extraction(r, "chief_complaint", "second value")
    assert r.chief_complaint == "second value"


def test_cc_does_not_affect_other_fields():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "cough")
    assert r.symptom_duration == ""
    assert r.symptom_severity == 0
    assert r.associated_symptoms == []


def test_cc_boolean_true_coerced():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", True)
    assert r.chief_complaint == "True"


def test_cc_boolean_false_coerced():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", False)
    assert r.chief_complaint == "False"


def test_cc_list_coerced_to_str():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", ["pain", "swelling"])
    assert isinstance(r.chief_complaint, str)


def test_cc_special_chars():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "pain (sharp) @ rest")
    assert r.chief_complaint == "pain (sharp) @ rest"


def test_cc_newline_in_value():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "chest pain\nnausea")
    assert "chest pain" in r.chief_complaint


# ---------------------------------------------------------------------------
# GROUP 2 — symptom_duration (10 tests)
# ---------------------------------------------------------------------------

def test_sd_three_days():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "3 days")
    assert r.symptom_duration == "3 days"


def test_sd_two_weeks():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "two weeks")
    assert r.symptom_duration == "two weeks"


def test_sd_since_yesterday():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "since yesterday")
    assert r.symptom_duration == "since yesterday"


def test_sd_empty_string():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "")
    assert r.symptom_duration == ""


def test_sd_number_coerced():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", 7)
    assert r.symptom_duration == "7"


def test_sd_overwrite():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "1 day")
    _apply_extraction(r, "symptom_duration", "1 week")
    assert r.symptom_duration == "1 week"


def test_sd_unicode():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "desde ayer")
    assert r.symptom_duration == "desde ayer"


def test_sd_does_not_affect_chief_complaint():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "3 days")
    assert r.chief_complaint == ""


def test_sd_long_string():
    r = IntakeRecord()
    val = "approximately three to four weeks give or take a few days"
    _apply_extraction(r, "symptom_duration", val)
    assert r.symptom_duration == val


def test_sd_whitespace_only():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "  ")
    assert r.symptom_duration == "  "


# ---------------------------------------------------------------------------
# GROUP 3 — symptom_severity (20 tests)
# ---------------------------------------------------------------------------

def test_severity_int_1():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 1)
    assert r.symptom_severity == 1


def test_severity_int_2():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 2)
    assert r.symptom_severity == 2


def test_severity_int_3():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 3)
    assert r.symptom_severity == 3


def test_severity_int_4():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 4)
    assert r.symptom_severity == 4


def test_severity_int_5():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 5)
    assert r.symptom_severity == 5


def test_severity_int_6():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 6)
    assert r.symptom_severity == 6


def test_severity_int_7():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 7)
    assert r.symptom_severity == 7


def test_severity_int_8():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 8)
    assert r.symptom_severity == 8


def test_severity_int_9():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 9)
    assert r.symptom_severity == 9


def test_severity_int_10():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 10)
    assert r.symptom_severity == 10


def test_severity_float_truncated():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 7.5)
    assert r.symptom_severity == 7


def test_severity_string_digit():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", "8")
    assert r.symptom_severity == 8


def test_severity_string_word_fails_gracefully():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", "eight")
    assert r.symptom_severity == 0


def test_severity_none_noop():
    r = IntakeRecord()
    r.symptom_severity = 5
    _apply_extraction(r, "symptom_severity", None)
    assert r.symptom_severity == 5


def test_severity_zero_stays_zero():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 0)
    assert r.symptom_severity == 0


def test_severity_eleven_stored_as_is():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 11)
    assert r.symptom_severity == 11


def test_severity_negative_stored():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", -1)
    assert r.symptom_severity == -1


def test_severity_overwrite():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 3)
    _apply_extraction(r, "symptom_severity", 9)
    assert r.symptom_severity == 9


def test_severity_string_float_stays_zero():
    # int("6.9") raises ValueError — code catches it and leaves severity=0
    # To convert a float string, caller must pass int(float("6.9")) upstream
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", "6.9")
    assert r.symptom_severity == 0


def test_severity_does_not_affect_chief_complaint():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 5)
    assert r.chief_complaint == ""


# ---------------------------------------------------------------------------
# GROUP 4 — associated_symptoms (15 tests)
# ---------------------------------------------------------------------------

def test_assoc_list_input_extends():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["nausea", "dizziness"])
    assert "nausea" in r.associated_symptoms
    assert "dizziness" in r.associated_symptoms


def test_assoc_single_string_wraps_to_list():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", "fatigue")
    assert "fatigue" in r.associated_symptoms
    assert isinstance(r.associated_symptoms, list)


def test_assoc_empty_list_extends_with_nothing():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", [])
    assert r.associated_symptoms == []


def test_assoc_none_is_noop():
    r = IntakeRecord()
    r.associated_symptoms = ["existing"]
    _apply_extraction(r, "associated_symptoms", None)
    assert r.associated_symptoms == ["existing"]


def test_assoc_multiple_calls_accumulate():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["nausea"])
    _apply_extraction(r, "associated_symptoms", ["vomiting"])
    assert "nausea" in r.associated_symptoms
    assert "vomiting" in r.associated_symptoms
    assert len(r.associated_symptoms) == 2


def test_assoc_three_accumulation_calls():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["a"])
    _apply_extraction(r, "associated_symptoms", ["b"])
    _apply_extraction(r, "associated_symptoms", ["c"])
    assert r.associated_symptoms == ["a", "b", "c"]


def test_assoc_alias_associated():
    r = IntakeRecord()
    _apply_extraction(r, "associated", ["sweating"])
    assert "sweating" in r.associated_symptoms


def test_assoc_does_not_affect_medications():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["rash"])
    assert r.current_medications == []


def test_assoc_unicode_symptom():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["fi\u00e8vre"])
    assert "fi\u00e8vre" in r.associated_symptoms


def test_assoc_list_of_one():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["headache"])
    assert r.associated_symptoms == ["headache"]


def test_assoc_large_list():
    r = IntakeRecord()
    symptoms = [f"symptom_{i}" for i in range(20)]
    _apply_extraction(r, "associated_symptoms", symptoms)
    assert len(r.associated_symptoms) == 20


def test_assoc_duplicate_values_both_stored():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["pain"])
    _apply_extraction(r, "associated_symptoms", ["pain"])
    assert len(r.associated_symptoms) == 2


def test_assoc_string_with_comma_not_split():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", "nausea, vomiting")
    assert len(r.associated_symptoms) == 1
    assert r.associated_symptoms[0] == "nausea, vomiting"


def test_assoc_integer_value_wrapped():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", 99)
    assert isinstance(r.associated_symptoms, list)


def test_assoc_initial_state_empty():
    r = IntakeRecord()
    assert r.associated_symptoms == []


# ---------------------------------------------------------------------------
# GROUP 5 — medical_history and alias "history" (10 tests)
# ---------------------------------------------------------------------------

def test_history_list_field():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", ["hypertension", "diabetes"])
    assert "hypertension" in r.medical_history
    assert "diabetes" in r.medical_history


def test_history_alias_works():
    r = IntakeRecord()
    _apply_extraction(r, "history", ["asthma"])
    assert "asthma" in r.medical_history


def test_history_single_string():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", "GERD")
    assert "GERD" in r.medical_history


def test_history_alias_single_string():
    r = IntakeRecord()
    _apply_extraction(r, "history", "depression")
    assert "depression" in r.medical_history


def test_history_accumulates():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", ["heart disease"])
    _apply_extraction(r, "history", ["stroke"])
    assert "heart disease" in r.medical_history
    assert "stroke" in r.medical_history


def test_history_none_is_noop():
    r = IntakeRecord()
    r.medical_history = ["existing"]
    _apply_extraction(r, "medical_history", None)
    assert r.medical_history == ["existing"]


def test_history_empty_list():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", [])
    assert r.medical_history == []


def test_history_does_not_affect_medications():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", ["cancer"])
    assert r.current_medications == []


def test_history_unicode():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", ["\u00e9nfermedad cr\u00f3nica"])
    assert "\u00e9nfermedad cr\u00f3nica" in r.medical_history


def test_history_initial_empty():
    r = IntakeRecord()
    assert r.medical_history == []


# ---------------------------------------------------------------------------
# GROUP 6 — current_medications and alias "medications" (10 tests)
# ---------------------------------------------------------------------------

def test_meds_list_field():
    r = IntakeRecord()
    _apply_extraction(r, "current_medications", ["aspirin", "lisinopril"])
    assert "aspirin" in r.current_medications
    assert "lisinopril" in r.current_medications


def test_meds_alias_works():
    r = IntakeRecord()
    _apply_extraction(r, "medications", ["metformin"])
    assert "metformin" in r.current_medications


def test_meds_single_string():
    r = IntakeRecord()
    _apply_extraction(r, "current_medications", "warfarin")
    assert "warfarin" in r.current_medications


def test_meds_alias_single_string():
    r = IntakeRecord()
    _apply_extraction(r, "medications", "atorvastatin")
    assert "atorvastatin" in r.current_medications


def test_meds_accumulate_across_calls():
    r = IntakeRecord()
    _apply_extraction(r, "current_medications", ["aspirin"])
    _apply_extraction(r, "medications", ["ibuprofen"])
    assert "aspirin" in r.current_medications
    assert "ibuprofen" in r.current_medications


def test_meds_none_is_noop():
    r = IntakeRecord()
    r.current_medications = ["existing"]
    _apply_extraction(r, "current_medications", None)
    assert r.current_medications == ["existing"]


def test_meds_empty_list():
    r = IntakeRecord()
    _apply_extraction(r, "current_medications", [])
    assert r.current_medications == []


def test_meds_does_not_affect_allergies():
    r = IntakeRecord()
    _apply_extraction(r, "current_medications", ["aspirin"])
    assert r.allergies == []


def test_meds_unicode():
    r = IntakeRecord()
    _apply_extraction(r, "medications", ["parac\u00e9tamol"])
    assert "parac\u00e9tamol" in r.current_medications


def test_meds_initial_empty():
    r = IntakeRecord()
    assert r.current_medications == []


# ---------------------------------------------------------------------------
# GROUP 7 — allergies (10 tests)
# ---------------------------------------------------------------------------

def test_allergies_list():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", ["penicillin", "sulfa"])
    assert "penicillin" in r.allergies
    assert "sulfa" in r.allergies


def test_allergies_single_string():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", "latex")
    assert "latex" in r.allergies


def test_allergies_accumulate_three_calls():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", ["nuts"])
    _apply_extraction(r, "allergies", ["shellfish"])
    _apply_extraction(r, "allergies", ["pollen"])
    assert "nuts" in r.allergies
    assert "shellfish" in r.allergies
    assert "pollen" in r.allergies
    assert len(r.allergies) == 3


def test_allergies_none_is_noop():
    r = IntakeRecord()
    r.allergies = ["existing"]
    _apply_extraction(r, "allergies", None)
    assert r.allergies == ["existing"]


def test_allergies_empty_list():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", [])
    assert r.allergies == []


def test_allergies_does_not_affect_medications():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", ["aspirin"])
    assert r.current_medications == []


def test_allergies_unicode():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", ["p\u00e9nicilline"])
    assert "p\u00e9nicilline" in r.allergies


def test_allergies_initial_empty():
    r = IntakeRecord()
    assert r.allergies == []


def test_allergies_large_list():
    r = IntakeRecord()
    allergens = [f"allergen_{i}" for i in range(10)]
    _apply_extraction(r, "allergies", allergens)
    assert len(r.allergies) == 10


def test_allergies_two_call_total_count():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", ["a", "b"])
    _apply_extraction(r, "allergies", ["c"])
    assert len(r.allergies) == 3


# ---------------------------------------------------------------------------
# GROUP 8 — language (5 tests)
# ---------------------------------------------------------------------------

def test_language_es():
    r = IntakeRecord()
    _apply_extraction(r, "language", "es")
    assert r.language == "es"


def test_language_fr():
    r = IntakeRecord()
    _apply_extraction(r, "language", "fr")
    assert r.language == "fr"


def test_language_en():
    r = IntakeRecord()
    _apply_extraction(r, "language", "en")
    assert r.language == "en"


def test_language_overwrite():
    r = IntakeRecord()
    _apply_extraction(r, "language", "es")
    _apply_extraction(r, "language", "fr")
    assert r.language == "fr"


def test_language_does_not_affect_chief_complaint():
    r = IntakeRecord()
    _apply_extraction(r, "language", "de")
    assert r.chief_complaint == ""


# ---------------------------------------------------------------------------
# GROUP 9 — None field_name (5 tests)
# ---------------------------------------------------------------------------

def test_none_field_name_noop_chief():
    r = IntakeRecord()
    r.chief_complaint = "existing"
    _apply_extraction(r, None, "chest pain")
    assert r.chief_complaint == "existing"


def test_none_field_name_noop_severity():
    r = IntakeRecord()
    r.symptom_severity = 5
    _apply_extraction(r, None, 8)
    assert r.symptom_severity == 5


def test_none_field_name_noop_medications():
    r = IntakeRecord()
    r.current_medications = ["aspirin"]
    _apply_extraction(r, None, ["ibuprofen"])
    assert r.current_medications == ["aspirin"]


def test_none_field_name_record_unchanged():
    r = IntakeRecord()
    _apply_extraction(r, None, "some value")
    assert r.chief_complaint == ""
    assert r.symptom_duration == ""
    assert r.symptom_severity == 0
    assert r.associated_symptoms == []
    assert r.medical_history == []
    assert r.current_medications == []
    assert r.allergies == []


def test_none_field_name_none_value_also_noop():
    r = IntakeRecord()
    r.chief_complaint = "cough"
    _apply_extraction(r, None, None)
    assert r.chief_complaint == "cough"


# ---------------------------------------------------------------------------
# GROUP 10 — None value (5 tests)
# ---------------------------------------------------------------------------

def test_none_value_chief_complaint_noop():
    r = IntakeRecord()
    r.chief_complaint = "existing"
    _apply_extraction(r, "chief_complaint", None)
    assert r.chief_complaint == "existing"


def test_none_value_severity_noop():
    r = IntakeRecord()
    r.symptom_severity = 7
    _apply_extraction(r, "symptom_severity", None)
    assert r.symptom_severity == 7


def test_none_value_medications_noop():
    r = IntakeRecord()
    r.current_medications = ["aspirin"]
    _apply_extraction(r, "current_medications", None)
    assert r.current_medications == ["aspirin"]


def test_none_value_allergies_noop():
    r = IntakeRecord()
    r.allergies = ["latex"]
    _apply_extraction(r, "allergies", None)
    assert r.allergies == ["latex"]


def test_none_value_language_noop():
    r = IntakeRecord()
    r.language = "es"
    _apply_extraction(r, "language", None)
    assert r.language == "es"


# ---------------------------------------------------------------------------
# GROUP 11 — Unknown field names (15 tests)
# ---------------------------------------------------------------------------

def test_unknown_blood_pressure_noop():
    r = IntakeRecord()
    _apply_extraction(r, "blood_pressure", "120/80")
    assert r.chief_complaint == ""
    assert r.associated_symptoms == []


def test_unknown_temperature_noop():
    r = IntakeRecord()
    _apply_extraction(r, "temperature", "101.3F")
    assert r.chief_complaint == ""


def test_unknown_chief_complaint_description_noop():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint_description", "persistent pain")
    assert r.chief_complaint == ""


def test_uppercase_chief_complaint_is_lowercased_and_applied():
    # _apply_extraction does field_name.lower().strip(), so "CHIEF_COMPLAINT"
    # becomes "chief_complaint" and IS applied — uppercase is NOT a no-op.
    r = IntakeRecord()
    _apply_extraction(r, "CHIEF_COMPLAINT", "headache")
    assert r.chief_complaint == "headache"


def test_unknown_severity_alias_noop():
    r = IntakeRecord()
    _apply_extraction(r, "severity", 5)
    assert r.symptom_severity == 0


def test_unknown_complaint_noop():
    r = IntakeRecord()
    _apply_extraction(r, "complaint", "chest pain")
    assert r.chief_complaint == ""


def test_unknown_dx_noop():
    r = IntakeRecord()
    _apply_extraction(r, "dx", "myocardial infarction")
    assert r.medical_history == []


def test_unknown_rx_noop():
    r = IntakeRecord()
    _apply_extraction(r, "rx", ["aspirin"])
    assert r.current_medications == []


def test_unknown_symptoms_alias_noop():
    r = IntakeRecord()
    _apply_extraction(r, "symptoms", ["fever"])
    assert r.associated_symptoms == []


def test_unknown_duration_noop():
    r = IntakeRecord()
    _apply_extraction(r, "duration", "3 days")
    assert r.symptom_duration == ""


def test_unknown_pain_scale_noop():
    r = IntakeRecord()
    _apply_extraction(r, "pain_scale", 8)
    assert r.symptom_severity == 0


def test_unknown_note_noop():
    r = IntakeRecord()
    _apply_extraction(r, "note", "patient appears anxious")
    assert r.chief_complaint == ""


def test_unknown_empty_string_field_noop():
    r = IntakeRecord()
    _apply_extraction(r, "", "value")
    assert r.chief_complaint == ""
    assert r.symptom_duration == ""


def test_unknown_whitespace_field_noop():
    r = IntakeRecord()
    _apply_extraction(r, "   ", "value")
    assert r.chief_complaint == ""


def test_unknown_field_record_fully_unchanged():
    r = IntakeRecord()
    _apply_extraction(r, "unknown_field_xyz", "some value")
    assert r.chief_complaint == ""
    assert r.symptom_duration == ""
    assert r.symptom_severity == 0
    assert r.associated_symptoms == []
    assert r.medical_history == []
    assert r.current_medications == []
    assert r.allergies == []
    assert r.language == "en"


# ---------------------------------------------------------------------------
# GROUP 12 — Accumulation behavior (10 tests)
# ---------------------------------------------------------------------------

def test_accum_associated_extends_not_replaces():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["nausea"])
    _apply_extraction(r, "associated_symptoms", ["vomiting"])
    assert len(r.associated_symptoms) == 2
    assert "nausea" in r.associated_symptoms
    assert "vomiting" in r.associated_symptoms


def test_accum_medical_history_extends():
    r = IntakeRecord()
    _apply_extraction(r, "medical_history", ["diabetes"])
    _apply_extraction(r, "medical_history", ["hypertension"])
    assert len(r.medical_history) == 2


def test_accum_medications_extends():
    r = IntakeRecord()
    _apply_extraction(r, "current_medications", ["aspirin"])
    _apply_extraction(r, "current_medications", ["metformin"])
    assert len(r.current_medications) == 2


def test_accum_allergies_extends():
    r = IntakeRecord()
    _apply_extraction(r, "allergies", ["penicillin"])
    _apply_extraction(r, "allergies", ["sulfa"])
    assert len(r.allergies) == 2


def test_accum_chief_complaint_overwrites():
    r = IntakeRecord()
    _apply_extraction(r, "chief_complaint", "first")
    _apply_extraction(r, "chief_complaint", "second")
    assert r.chief_complaint == "second"


def test_accum_symptom_duration_overwrites():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_duration", "1 day")
    _apply_extraction(r, "symptom_duration", "2 weeks")
    assert r.symptom_duration == "2 weeks"


def test_accum_symptom_severity_overwrites():
    r = IntakeRecord()
    _apply_extraction(r, "symptom_severity", 3)
    _apply_extraction(r, "symptom_severity", 9)
    assert r.symptom_severity == 9


def test_accum_language_overwrites():
    r = IntakeRecord()
    _apply_extraction(r, "language", "en")
    _apply_extraction(r, "language", "es")
    assert r.language == "es"


def test_accum_mixed_list_and_string():
    r = IntakeRecord()
    _apply_extraction(r, "associated_symptoms", ["nausea"])
    _apply_extraction(r, "associated_symptoms", "fever")
    assert "nausea" in r.associated_symptoms
    assert "fever" in r.associated_symptoms
    assert len(r.associated_symptoms) == 2


def test_accum_five_medication_calls():
    r = IntakeRecord()
    meds = ["aspirin", "metformin", "lisinopril", "atorvastatin", "omeprazole"]
    for med in meds:
        _apply_extraction(r, "current_medications", [med])
    assert len(r.current_medications) == 5
    for med in meds:
        assert med in r.current_medications


# ---------------------------------------------------------------------------
# GROUP 13 — IntakeRecord.is_ready_for_clinician() (10 tests)
# ---------------------------------------------------------------------------

def test_ready_true_when_both_set():
    r = IntakeRecord()
    r.chief_complaint = "headache"
    r.symptom_duration = "3 days"
    assert r.is_ready_for_clinician() is True


def test_ready_false_when_both_empty():
    r = IntakeRecord()
    assert r.is_ready_for_clinician() is False


def test_ready_false_when_only_complaint():
    r = IntakeRecord()
    r.chief_complaint = "chest pain"
    assert r.is_ready_for_clinician() is False


def test_ready_false_when_only_duration():
    r = IntakeRecord()
    r.symptom_duration = "2 weeks"
    assert r.is_ready_for_clinician() is False


def test_ready_true_with_whitespace_values():
    r = IntakeRecord()
    r.chief_complaint = "   "
    r.symptom_duration = "  "
    assert r.is_ready_for_clinician() is True


def test_ready_false_after_complaint_cleared():
    r = IntakeRecord()
    r.chief_complaint = "pain"
    r.symptom_duration = "1 day"
    r.chief_complaint = ""
    assert r.is_ready_for_clinician() is False


def test_ready_false_after_duration_cleared():
    r = IntakeRecord()
    r.chief_complaint = "pain"
    r.symptom_duration = "1 day"
    r.symptom_duration = ""
    assert r.is_ready_for_clinician() is False


def test_ready_true_with_unicode_values():
    r = IntakeRecord()
    r.chief_complaint = "dolor de cabeza"
    r.symptom_duration = "tres d\u00edas"
    assert r.is_ready_for_clinician() is True


def test_ready_returns_bool():
    r = IntakeRecord()
    r.chief_complaint = "cough"
    r.symptom_duration = "1 week"
    result = r.is_ready_for_clinician()
    assert isinstance(result, bool)


def test_ready_fresh_record_is_false():
    r = IntakeRecord()
    assert r.is_ready_for_clinician() is False


# ---------------------------------------------------------------------------
# GROUP 14 — IntakeRecord.to_provider_summary() (10 tests)
# ---------------------------------------------------------------------------

def test_summary_contains_chief_complaint_header():
    r = IntakeRecord()
    r.chief_complaint = "back pain"
    summary = r.to_provider_summary()
    assert "CHIEF COMPLAINT" in summary.upper() or "chief complaint" in summary.lower()


def test_summary_contains_chief_complaint_value():
    r = IntakeRecord()
    r.chief_complaint = "severe migraine"
    summary = r.to_provider_summary()
    assert "severe migraine" in summary


def test_summary_contains_medication_name():
    r = IntakeRecord()
    r.chief_complaint = "chest pain"
    r.symptom_duration = "1 day"
    r.current_medications = ["warfarin", "aspirin"]
    summary = r.to_provider_summary()
    assert "warfarin" in summary or "aspirin" in summary


def test_summary_contains_allergy_name():
    r = IntakeRecord()
    r.chief_complaint = "rash"
    r.symptom_duration = "3 days"
    r.allergies = ["penicillin"]
    summary = r.to_provider_summary()
    assert "penicillin" in summary


def test_summary_contains_red_flag_section():
    r = IntakeRecord()
    r.chief_complaint = "difficulty breathing"
    r.symptom_duration = "1 hour"
    r.red_flags = ["possible pulmonary embolism"]
    summary = r.to_provider_summary()
    assert "RED FLAG" in summary.upper() or "red flag" in summary.lower()


def test_summary_contains_severity_value():
    r = IntakeRecord()
    r.chief_complaint = "pain"
    r.symptom_duration = "2 days"
    r.symptom_severity = 8
    summary = r.to_provider_summary()
    assert "8" in summary


def test_summary_not_rated_or_zero_when_severity_zero():
    r = IntakeRecord()
    r.chief_complaint = "cough"
    r.symptom_duration = "1 week"
    r.symptom_severity = 0
    summary = r.to_provider_summary()
    assert "not rated" in summary.lower() or "0" in summary


def test_summary_returns_string():
    r = IntakeRecord()
    summary = r.to_provider_summary()
    assert isinstance(summary, str)


def test_summary_non_empty():
    r = IntakeRecord()
    r.chief_complaint = "fever"
    r.symptom_duration = "2 days"
    summary = r.to_provider_summary()
    assert len(summary) > 0


def test_summary_multiple_medications_at_least_one_present():
    r = IntakeRecord()
    r.chief_complaint = "fatigue"
    r.symptom_duration = "1 month"
    r.current_medications = ["metformin", "lisinopril", "atorvastatin"]
    summary = r.to_provider_summary()
    found = sum(1 for m in ["metformin", "lisinopril", "atorvastatin"] if m in summary)
    assert found >= 1
