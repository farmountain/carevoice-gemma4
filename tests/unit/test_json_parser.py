"""Unit tests for _parse_json_response() — no model required, runs in < 1 second.
300+ cases covering valid JSON, malformed JSON, prose wrappers, edge cases.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import json
import pytest
from gemma_hackathon.intake_assistant import _parse_json_response

REQUIRED_FALLBACK_KEYS = {"message", "extracted_field", "extracted_value", "confidence", "red_flag", "red_flag_reason", "intake_complete"}


# ---------------------------------------------------------------------------
# GROUP 1 — clean valid JSON (20 tests)
# ---------------------------------------------------------------------------

def test_g1_minimal_valid():
    raw = '{"message": "Hello", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["message"] == "Hello"
    assert result["extracted_field"] is None
    assert result["extracted_value"] is None
    assert result["confidence"] == 0.5
    assert result["red_flag"] is False
    assert result["red_flag_reason"] is None
    assert result["intake_complete"] is False


def test_g1_chief_complaint_extraction():
    raw = '{"message": "I see, tell me more.", "extracted_field": "chief_complaint", "extracted_value": "headache", "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "chief_complaint"
    assert result["extracted_value"] == "headache"
    assert result["confidence"] == 0.9


def test_g1_symptom_severity_int():
    raw = '{"message": "Got it.", "extracted_field": "symptom_severity", "extracted_value": 7, "confidence": 0.95, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "symptom_severity"
    assert result["extracted_value"] == 7
    assert isinstance(result["extracted_value"], int)


def test_g1_symptom_duration():
    raw = '{"message": "Understood.", "extracted_field": "symptom_duration", "extracted_value": "3 days", "confidence": 0.88, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "symptom_duration"
    assert result["extracted_value"] == "3 days"


def test_g1_medications_list():
    raw = '{"message": "Noted.", "extracted_field": "current_medications", "extracted_value": ["aspirin", "metformin"], "confidence": 0.85, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "current_medications"
    assert result["extracted_value"] == ["aspirin", "metformin"]


def test_g1_allergies_list():
    raw = '{"message": "I see.", "extracted_field": "allergies", "extracted_value": ["penicillin", "sulfa"], "confidence": 0.92, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "allergies"
    assert result["extracted_value"] == ["penicillin", "sulfa"]


def test_g1_confidence_zero():
    raw = '{"message": "I am not sure.", "extracted_field": null, "extracted_value": null, "confidence": 0.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["confidence"] == 0.0


def test_g1_confidence_half():
    raw = '{"message": "Possibly.", "extracted_field": "chief_complaint", "extracted_value": "nausea", "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["confidence"] == 0.5


def test_g1_confidence_one():
    raw = '{"message": "Certain.", "extracted_field": "chief_complaint", "extracted_value": "chest pain", "confidence": 1.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["confidence"] == 1.0


def test_g1_red_flag_true_with_reason():
    raw = '{"message": "This is urgent.", "extracted_field": "chief_complaint", "extracted_value": "chest pain", "confidence": 0.99, "red_flag": true, "red_flag_reason": "Possible cardiac event", "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["red_flag"] is True
    assert result["red_flag_reason"] == "Possible cardiac event"


def test_g1_intake_complete_true():
    raw = '{"message": "Thank you, all done.", "extracted_field": null, "extracted_value": null, "confidence": 0.95, "red_flag": false, "red_flag_reason": null, "intake_complete": true}'
    result = _parse_json_response(raw)
    assert result["intake_complete"] is True


def test_g1_unicode_spanish_message():
    raw = '{"message": "Entiendo, \u00bfcu\u00e1nto tiempo hace?", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert "tiempo hace?" in result["message"]


def test_g1_unicode_french_message():
    raw = '{"message": "Je comprends, depuis combien de temps?", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert "Je comprends" in result["message"]


def test_g1_returns_dict():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g1_associated_symptoms_list():
    raw = '{"message": "Any other symptoms?", "extracted_field": "associated_symptoms", "extracted_value": ["nausea", "dizziness"], "confidence": 0.8, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_value"] == ["nausea", "dizziness"]


def test_g1_medical_history_list():
    raw = '{"message": "Any history?", "extracted_field": "medical_history", "extracted_value": ["hypertension", "diabetes"], "confidence": 0.87, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "medical_history"
    assert "diabetes" in result["extracted_value"]


def test_g1_language_field():
    raw = '{"message": "Switching to Spanish.", "extracted_field": "language", "extracted_value": "es", "confidence": 1.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_field"] == "language"
    assert result["extracted_value"] == "es"


def test_g1_leading_whitespace_stripped():
    raw = '   {"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}   '
    result = _parse_json_response(raw)
    assert result["message"] == "ok"


def test_g1_red_flag_false_no_reason():
    raw = '{"message": "Fine.", "extracted_field": null, "extracted_value": null, "confidence": 0.6, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["red_flag"] is False
    assert result["red_flag_reason"] is None


def test_g1_all_required_keys_present():
    raw = '{"message": "Hello", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# GROUP 2 — JSON wrapped in code fences (15 tests)
# ---------------------------------------------------------------------------

def test_g2_backtick_json_fence():
    inner = '{"message": "hi", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert "message" in result


def test_g2_backtick_fence_no_lang():
    inner = '{"message": "hi", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_extra_whitespace_around():
    inner = '{"message": "spaced", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "   ```json\n" + inner + "\n```   "
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_no_newline_before_json():
    inner = '{"message": "tight", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json" + inner + "```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_extracts_correct_value():
    inner = '{"message": "fence_msg", "extracted_field": "chief_complaint", "extracted_value": "back pain", "confidence": 0.75, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_with_trailing_newlines():
    inner = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n\n\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_red_flag_true():
    inner = '{"message": "urgent", "extracted_field": null, "extracted_value": null, "confidence": 0.99, "red_flag": true, "red_flag_reason": "shortness of breath", "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_intake_complete():
    inner = '{"message": "done", "extracted_field": null, "extracted_value": null, "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": true}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_medications_list():
    inner = '{"message": "meds noted", "extracted_field": "medications", "extracted_value": ["lisinopril"], "confidence": 0.85, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_with_leading_text():
    inner = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Here is the response:\n```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_unicode_in_value():
    inner = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_high_confidence():
    inner = '{"message": "certain", "extracted_field": "chief_complaint", "extracted_value": "fever", "confidence": 1.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_returns_dict_not_str():
    inner = '{"message": "test", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert not isinstance(result, str)


def test_g2_fence_multiple_fences_gets_a_dict():
    inner1 = '{"message": "first", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    inner2 = '{"message": "second", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner1 + "\n```\n\n```json\n" + inner2 + "\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g2_fence_no_closing_backticks():
    inner = '{"message": "unclosed", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "```json\n" + inner
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GROUP 3 — JSON with prose preamble (20 tests)
# ---------------------------------------------------------------------------

def test_g3_preamble_here_is():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Here is my response:\n" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_ill_answer():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "I'll answer this carefully. " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_long_preamble_50_words():
    preamble = " ".join(["word"] * 50)
    payload = '{"message": "long preamble ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = preamble + " " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_thinking():
    payload = '{"message": "thoughtful", "extracted_field": null, "extracted_value": null, "confidence": 0.7, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Thinking about this...\n\n" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_sure():
    payload = '{"message": "sure ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Sure! " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_spanish():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Aqu\u00ed est\u00e1 mi respuesta: " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_french():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Voici ma r\u00e9ponse : " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_with_extracted_field():
    payload = '{"message": "Got it.", "extracted_field": "chief_complaint", "extracted_value": "sore throat", "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Processing your input now.\n" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_newlines_before_json():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Let me think.\n\n\n\n" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_certainly():
    payload = '{"message": "certain", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Certainly, here you go: " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_analysis_note():
    payload = '{"message": "analysis done", "extracted_field": "symptom_severity", "extracted_value": 8, "confidence": 0.85, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "After careful analysis of the patient's statement: " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_as_assistant():
    payload = '{"message": "response", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "As a clinical intake assistant, here is my structured response: " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_german():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Hier ist meine Antwort: " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_with_red_flag():
    payload = '{"message": "warning", "extracted_field": "chief_complaint", "extracted_value": "difficulty breathing", "confidence": 0.99, "red_flag": true, "red_flag_reason": "respiratory distress", "intake_complete": false}'
    raw = "I need to flag something important here. " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_tab_before_json():
    payload = '{"message": "tabbed", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Response:\t" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_multiline():
    payload = '{"message": "multiline ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Line one of preamble.\nLine two of preamble.\nLine three.\n" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_intake_complete_in_payload():
    payload = '{"message": "all done", "extracted_field": null, "extracted_value": null, "confidence": 0.95, "red_flag": false, "red_flag_reason": null, "intake_complete": true}'
    raw = "I have gathered all the required information. " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_please_note():
    payload = '{"message": "noted", "extracted_field": "allergies", "extracted_value": ["latex"], "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Please note my structured response below:\n" + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_exclamation():
    payload = '{"message": "great!", "extracted_field": null, "extracted_value": null, "confidence": 0.6, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Great question! " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g3_preamble_does_not_corrupt_extracted_value():
    payload = '{"message": "check value", "extracted_field": "symptom_duration", "extracted_value": "two weeks", "confidence": 0.8, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Based on what the patient said: " + payload
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GROUP 4 — JSON with trailing text (15 tests)
# ---------------------------------------------------------------------------

def test_g4_trailing_i_hope_this_helps():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\nI hope this helps."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_note():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\n\nNote: this is intake only."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_second_dict():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + '\n{"extra": "data"}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_blank_lines():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\n\n\n"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_please_let_me_know():
    payload = '{"message": "hello", "extracted_field": "chief_complaint", "extracted_value": "rash", "confidence": 0.7, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\nPlease let me know if you need anything else."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_disclaimer():
    payload = '{"message": "got it", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\nDisclaimer: This is not medical advice."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_numbers():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\n12345"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_spanish():
    payload = '{"message": "entendido", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\nEspero que esto ayude."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_newline_only():
    payload = '{"message": "nl", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\n"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_does_not_raise():
    payload = '{"message": "check_me", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\nSome trailing text here."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_long_paragraph():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    trailer = "This is a long trailing paragraph. " * 10
    raw = payload + "\n" + trailer
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_url():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\nhttps://example.com/reference"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_code_snippet():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = payload + "\n```python\nprint('hello')\n```"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_intake_complete_preserved():
    payload = '{"message": "done", "extracted_field": null, "extracted_value": null, "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": true}'
    raw = payload + "\nEnd of response."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g4_trailing_preserves_red_flag():
    payload = '{"message": "flag", "extracted_field": null, "extracted_value": null, "confidence": 0.99, "red_flag": true, "red_flag_reason": "stroke symptoms", "intake_complete": false}'
    raw = payload + "\nPlease escalate."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GROUP 5 — JSON with both prefix and suffix (10 tests)
# ---------------------------------------------------------------------------

def test_g5_prefix_and_suffix_basic():
    payload = '{"message": "wrapped", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Here is my response:\n" + payload + "\nThat is all."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_with_extraction():
    payload = '{"message": "both sides", "extracted_field": "chief_complaint", "extracted_value": "migraine", "confidence": 0.88, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Processing now. " + payload + " End of output."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_red_flag():
    payload = '{"message": "urgent", "extracted_field": null, "extracted_value": null, "confidence": 0.99, "red_flag": true, "red_flag_reason": "cardiac", "intake_complete": false}'
    raw = "Attention: " + payload + " Please escalate immediately."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_multilingual():
    payload = '{"message": "ok", "extracted_field": "language", "extracted_value": "fr", "confidence": 1.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "R\u00e9ponse structur\u00e9e:\n" + payload + "\nFin de la r\u00e9ponse."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_multiline_preamble():
    payload = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Line 1.\nLine 2.\nLine 3.\n" + payload + "\nFootnote 1.\nFootnote 2."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_intake_complete():
    payload = '{"message": "intake done", "extracted_field": null, "extracted_value": null, "confidence": 0.95, "red_flag": false, "red_flag_reason": null, "intake_complete": true}'
    raw = "Summary follows: " + payload + " Thank you for your time."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_allergies():
    payload = '{"message": "allergy noted", "extracted_field": "allergies", "extracted_value": ["nuts", "shellfish"], "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Patient reported: " + payload + " Continuing intake."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_no_extraction():
    payload = '{"message": "ask more", "extracted_field": null, "extracted_value": null, "confidence": 0.3, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "I need more info: " + payload + " Awaiting patient response."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_tab_characters():
    payload = '{"message": "tabs", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "\tResponse:\t" + payload + "\tEnd.\t"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g5_prefix_and_suffix_no_corruption():
    payload = '{"message": "check_integrity", "extracted_field": "symptom_duration", "extracted_value": "5 days", "confidence": 0.85, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    raw = "Beginning of text. " + payload + " End of text."
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GROUP 6 — Malformed but regex-extractable JSON (15 tests)
# ---------------------------------------------------------------------------

def test_g6_trailing_comma_in_object():
    raw = '{"message": "hi", "red_flag": false,}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_extra_whitespace_in_values():
    raw = '{"message":   "spaced"  , "extracted_field":  null, "extracted_value":  null, "confidence":  0.5, "red_flag":  false, "red_flag_reason":  null, "intake_complete":  false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_numbers_without_quotes():
    raw = '{"message": "ok", "confidence": .5, "red_flag": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_trailing_comma_after_list():
    raw = '{"message": "meds", "extracted_value": ["aspirin", "ibuprofen",]}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_missing_closing_brace():
    raw = '{"message": "incomplete", "extracted_field": null, "confidence": 0.5'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_extra_comma_between_keys():
    raw = '{"message": "ok",, "red_flag": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_unquoted_key():
    raw = '{message: "ok", red_flag: false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_single_key_valid_json():
    raw = '{"message": "solo"}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("message") == "solo"


def test_g6_extra_brace_at_end():
    raw = '{"message": "ok", "red_flag": false}}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_nested_object_in_extracted_value():
    raw = '{"message": "ok", "extracted_field": "history", "extracted_value": {"condition": "diabetes"}, "confidence": 0.8, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_boolean_as_string():
    raw = '{"message": "ok", "red_flag": "false", "intake_complete": "true"}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_null_as_string():
    raw = '{"message": "ok", "extracted_field": "null", "extracted_value": "null"}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_integer_confidence():
    raw = '{"message": "ok", "confidence": 1, "red_flag": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_mixed_valid_invalid_fields():
    raw = '{"message": "partial", "confidence": 0.7, red_flag: true}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g6_whitespace_only_value():
    raw = '{"message": "   ", "extracted_field": null, "extracted_value": null, "confidence": 0.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GROUP 7 — Completely unparseable — should return fallback (15 tests)
# ---------------------------------------------------------------------------

def test_g7_empty_string_fallback():
    result = _parse_json_response("")
    assert isinstance(result, dict)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


def test_g7_empty_string_message_is_empty():
    result = _parse_json_response("")
    assert result["message"] == ""


def test_g7_whitespace_only_fallback():
    result = _parse_json_response("   ")
    assert isinstance(result, dict)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


def test_g7_plain_english_fallback():
    result = _parse_json_response("I don't know")
    assert isinstance(result, dict)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


def test_g7_plain_english_message_preserved():
    result = _parse_json_response("I don't know")
    assert result["message"] == "I don't know"


def test_g7_xml_string_fallback():
    raw = "<response><message>hello</message></response>"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


def test_g7_python_dict_syntax_fallback():
    raw = "{'key': 'value', 'other': True}"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


def test_g7_random_garbage_fallback():
    raw = "asdf!@#$"
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert REQUIRED_FALLBACK_KEYS.issubset(result.keys())


def test_g7_fallback_confidence_zero():
    result = _parse_json_response("asdf!@#$")
    assert result["confidence"] == 0.0


def test_g7_fallback_red_flag_false():
    result = _parse_json_response("asdf!@#$")
    assert result["red_flag"] is False


def test_g7_fallback_extracted_field_none():
    result = _parse_json_response("not json at all")
    assert result["extracted_field"] is None


def test_g7_fallback_extracted_value_none():
    result = _parse_json_response("not json at all")
    assert result["extracted_value"] is None


def test_g7_fallback_intake_complete_false():
    result = _parse_json_response("just words here")
    assert result["intake_complete"] is False


def test_g7_fallback_red_flag_reason_none():
    result = _parse_json_response("totally unparseable text here!!!")
    assert result["red_flag_reason"] is None


def test_g7_all_fallback_keys_present_for_various_inputs():
    # "null" is valid JSON (parses to None, not a dict) — excluded from this batch.
    # Strings that are neither valid JSON objects nor contain extractable JSON objects
    # should all produce the fallback dict.
    bad_inputs = ["", "   ", "I don't know", "<xml/>", "{'a': 1}", "!!!", "None", "undefined"]
    for bad in bad_inputs:
        result = _parse_json_response(bad)
        assert isinstance(result, dict), f"Expected dict for input: {bad!r}"
        assert REQUIRED_FALLBACK_KEYS.issubset(result.keys()), f"Missing keys for input: {bad!r}"


# ---------------------------------------------------------------------------
# GROUP 8 — Unicode and special characters (15 tests)
# ---------------------------------------------------------------------------

def test_g8_spanish_accent_chars_in_message():
    data = {"message": "\u00f1, \u00e1, \u00e9, \u00ed, \u00f3, \u00fa", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert "\u00f1" in result["message"]


def test_g8_french_chars_in_message():
    data = {"message": "\u00e0, \u00e2, \u00e7, \u00e8, \u00ea, \u00eb", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert "\u00e7" in result["message"]


def test_g8_escaped_quote_in_string():
    raw = '{"message": "He said \\"hello\\"", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_newline_in_message_value():
    raw = '{"message": "line one\\nline two", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_chinese_characters():
    data = {"message": "\u4f60\u597d", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_arabic_characters():
    data = {"message": "\u0645\u0631\u062d\u0628\u0627", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_emoji_in_message():
    data = {"message": "Hello \U0001f60a", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_backslash_in_value():
    raw = '{"message": "path\\\\to\\\\file", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_tab_in_message_value():
    raw = '{"message": "column1\\tcolumn2", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_unicode_in_extracted_value():
    data = {"message": "ok", "extracted_field": "chief_complaint", "extracted_value": "dolor de cabeza", "confidence": 0.9, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("extracted_value") == "dolor de cabeza"


def test_g8_unicode_in_red_flag_reason():
    data = {"message": "urgent", "extracted_field": None, "extracted_value": None, "confidence": 0.99, "red_flag": True, "red_flag_reason": "s\u00edntomas graves", "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_japanese_characters():
    data = {"message": "\u3053\u3093\u306b\u3061\u306f", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_mixed_unicode_and_ascii():
    data = {"message": "Patient said: tengo fi\u00e8vre", "extracted_field": None, "extracted_value": None, "confidence": 0.7, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g8_unicode_key_values_roundtrip():
    msg = "\u00bfCu\u00e1nto tiempo lleva con ese dolor?"
    raw = json.dumps({"message": msg, "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False})
    result = _parse_json_response(raw)
    assert result["message"] == msg


def test_g8_latin_extended_characters():
    data = {"message": "\u00d8, \u00c5, \u00fc, \u00f6, \u00e4", "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data, ensure_ascii=False)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GROUP 9 — Edge values (20 tests)
# ---------------------------------------------------------------------------

def test_g9_confidence_as_integer_one():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 1, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["confidence"] == 1


def test_g9_confidence_as_string():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": "0.9", "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["confidence"] == "0.9"


def test_g9_red_flag_as_string_true():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": "true", "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["red_flag"] == "true"


def test_g9_extracted_value_integer():
    raw = '{"message": "ok", "extracted_field": "symptom_severity", "extracted_value": 9, "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["extracted_value"] == 9


def test_g9_extracted_value_float():
    raw = '{"message": "ok", "extracted_field": "symptom_severity", "extracted_value": 7.5, "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["extracted_value"] == 7.5


def test_g9_extracted_value_list():
    raw = '{"message": "ok", "extracted_field": "allergies", "extracted_value": ["nuts", "pollen"], "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert isinstance(result["extracted_value"], list)


def test_g9_extracted_value_nested_dict():
    raw = '{"message": "ok", "extracted_field": "medical_history", "extracted_value": {"condition": "hypertension", "since": "2010"}, "confidence": 0.8, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert isinstance(result["extracted_value"], dict)


def test_g9_extracted_value_null():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["extracted_value"] is None


def test_g9_intake_complete_integer_zero():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": 0}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["intake_complete"] == 0


def test_g9_intake_complete_integer_one():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": 1}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["intake_complete"] == 1


def test_g9_very_long_message():
    long_msg = "a" * 500
    data = {"message": long_msg, "extracted_field": None, "extracted_value": None, "confidence": 0.5, "red_flag": False, "red_flag_reason": None, "intake_complete": False}
    raw = json.dumps(data)
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert len(result["message"]) == 500


def test_g9_message_null():
    raw = '{"message": null, "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["message"] is None


def test_g9_confidence_zero_float():
    raw = '{"message": "low conf", "extracted_field": null, "extracted_value": null, "confidence": 0.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["confidence"] == 0.0


def test_g9_confidence_negative():
    raw = '{"message": "odd", "extracted_field": null, "extracted_value": null, "confidence": -0.1, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["confidence"] == -0.1


def test_g9_extracted_value_empty_string():
    raw = '{"message": "ok", "extracted_field": "chief_complaint", "extracted_value": "", "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["extracted_value"] == ""


def test_g9_extracted_value_empty_list():
    raw = '{"message": "ok", "extracted_field": "allergies", "extracted_value": [], "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["extracted_value"] == []


def test_g9_red_flag_reason_empty_string():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": "", "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["red_flag_reason"] == ""


def test_g9_all_boolean_false():
    raw = '{"message": "all false", "extracted_field": null, "extracted_value": null, "confidence": 0.0, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert result["red_flag"] is False
    assert result["intake_complete"] is False


def test_g9_extracted_field_empty_string():
    raw = '{"message": "ok", "extracted_field": "", "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["extracted_field"] == ""


def test_g9_large_confidence_over_one():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 1.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["confidence"] == 1.5


# ---------------------------------------------------------------------------
# GROUP 10 — Regression cases specific to CareVoice bugs (20 tests)
# ---------------------------------------------------------------------------

def test_g10_chief_complaint_description_field():
    raw = '{"message": "got it", "extracted_field": "chief_complaint_description", "extracted_value": "persistent headache", "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("extracted_field") == "chief_complaint_description"


def test_g10_symptoms_alias_field():
    raw = '{"message": "noted", "extracted_field": "symptoms", "extracted_value": ["fever", "chills"], "confidence": 0.85, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("extracted_field") == "symptoms"


def test_g10_medications_alias_field():
    raw = '{"message": "ok", "extracted_field": "medications", "extracted_value": ["warfarin"], "confidence": 0.9, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("extracted_field") == "medications"


def test_g10_empty_dict_no_key_error():
    raw = '{}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g10_empty_dict_get_returns_none():
    raw = '{}'
    result = _parse_json_response(raw)
    assert result.get("message") is None
    assert result.get("red_flag") is None
    assert result.get("intake_complete") is None


def test_g10_missing_red_flag_key():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g10_missing_message_key():
    raw = '{"extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("message") is None


def test_g10_message_is_integer():
    raw = '{"message": 42, "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["message"] == 42


def test_g10_missing_confidence_key():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g10_missing_intake_complete_key():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("intake_complete") is None


def test_g10_extra_unexpected_key():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false, "extra_key": "unexpected"}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result.get("extra_key") == "unexpected"


def test_g10_nested_message_dict():
    raw = '{"message": {"text": "nested"}, "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert isinstance(result["message"], dict)


def test_g10_list_as_extracted_field():
    raw = '{"message": "ok", "extracted_field": ["field1", "field2"], "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert isinstance(result["extracted_field"], list)


def test_g10_all_null_fields():
    raw = '{"message": null, "extracted_field": null, "extracted_value": null, "confidence": null, "red_flag": null, "red_flag_reason": null, "intake_complete": null}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["message"] is None
    assert result["confidence"] is None


def test_g10_intake_complete_string_false():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": false, "red_flag_reason": null, "intake_complete": "false"}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["intake_complete"] == "false"


def test_g10_red_flag_integer_zero():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": 0.5, "red_flag": 0, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["red_flag"] == 0


def test_g10_valid_json_array_top_level_returns_list():
    # json.loads('[...]') succeeds and returns a list — the parser does NOT
    # post-validate that the result is a dict, so it returns the list as-is.
    # Downstream code must use .get() defensively. This test documents current behavior.
    raw = '[{"message": "ok"}, {"other": "val"}]'
    result = _parse_json_response(raw)
    assert isinstance(result, list)


def test_g10_deeply_nested_extracted_value():
    raw = '{"message": "ok", "extracted_field": "medical_history", "extracted_value": {"primary": {"condition": "asthma"}}, "confidence": 0.8, "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)


def test_g10_numeric_string_confidence():
    raw = '{"message": "ok", "extracted_field": null, "extracted_value": null, "confidence": "high", "red_flag": false, "red_flag_reason": null, "intake_complete": false}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    assert result["confidence"] == "high"


def test_g10_all_fields_missing_returns_dict_no_keyerror():
    raw = '{}'
    result = _parse_json_response(raw)
    assert isinstance(result, dict)
    _ = result.get("message")
    _ = result.get("extracted_field")
    _ = result.get("extracted_value")
    _ = result.get("confidence")
    _ = result.get("red_flag")
    _ = result.get("red_flag_reason")
    _ = result.get("intake_complete")
