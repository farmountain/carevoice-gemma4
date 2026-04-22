"""
CareVoice Integration Tests — Stress Test Suite
================================================
Section A: Original 10 scenarios (regression gate) — mock, instant
Section B: Red flag category coverage — mock
Section C: Language coverage — mock
Section D: Extended corpus batch (500 scenarios) — mock by default, real model with --use-real-model
Section E: Edge case robustness — mock
Section F: Metrics summary report — mock, printed output

Run unit-only (fast):
    pytest tests/unit/ -v

Run all including extended corpus:
    pytest tests/ -v --max-scenarios=100

Run against real model (Kaggle GPU):
    pytest tests/ -v --use-real-model --max-scenarios=500
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import pytest
import statistics
from gemma_hackathon.intake_assistant import IntakeAssistant
from gemma_hackathon.scenarios import SAMPLE_SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scenario(sample_scenarios, scenario_id):
    """Return the scenario dict matching the given scenario_id."""
    for s in sample_scenarios:
        if s["scenario_id"] == scenario_id:
            return s
    raise KeyError(f"Scenario {scenario_id!r} not found in sample_scenarios")


def _run_subset(assistant, scenarios, max_n):
    """Run up to max_n scenarios and return list of (scenario, result) tuples."""
    subset = scenarios[:max_n]
    return [(s, assistant.run_scenario(s)) for s in subset]


# ===========================================================================
# SECTION A — Regression gate (10 original scenarios)
# ===========================================================================

class TestSectionA:
    """Original 10-scenario regression gate. Must all pass on every run."""

    def test_all_original_scenarios_pass(self, mock_assistant, sample_scenarios):
        """All 10 SAMPLE_SCENARIOS must satisfy D4 criterion with mock assistant."""
        results = [mock_assistant.run_scenario(s) for s in sample_scenarios]
        failed = [r.scenario_id for r in results if not r.passed_d4_criterion]
        print(f"\nSection A: {len(results) - len(failed)}/{len(results)} passed D4 criterion")
        assert len(failed) == 0, f"Scenarios failed D4 criterion: {failed}"

    def test_chief_complaint_captured_all_original(self, mock_assistant, sample_scenarios):
        """chief_complaint must be non-empty for all 10 original scenarios."""
        missing = []
        for scenario in sample_scenarios:
            result = mock_assistant.run_scenario(scenario)
            if not result.final_record.chief_complaint:
                missing.append(scenario["scenario_id"])
        assert len(missing) == 0, f"Missing chief_complaint in: {missing}"

    def test_red_flags_match_expected_original(self, mock_assistant, sample_scenarios):
        """bool(red_flags) must match expected_red_flag for all 10 scenarios."""
        mismatches = []
        for scenario in sample_scenarios:
            result = mock_assistant.run_scenario(scenario)
            actual = bool(result.final_record.red_flags)
            expected = scenario["expected_red_flag"]
            if actual != expected:
                mismatches.append({
                    "scenario_id": scenario["scenario_id"],
                    "expected": expected,
                    "actual": actual,
                    "red_flags": result.final_record.red_flags,
                })
        if mismatches:
            print(f"\nRed-flag mismatches: {mismatches}")
        assert len(mismatches) == 0, f"Red flag mismatches: {mismatches}"

    def test_all_turns_complete_original(self, mock_assistant, sample_scenarios):
        """Number of TurnResults must equal number of turns in the scenario definition."""
        mismatches = []
        for scenario in sample_scenarios:
            result = mock_assistant.run_scenario(scenario)
            expected_turns = len(scenario["turns"])
            actual_turns = len(result.turns)
            if expected_turns != actual_turns:
                mismatches.append({
                    "scenario_id": scenario["scenario_id"],
                    "expected": expected_turns,
                    "actual": actual_turns,
                })
        assert len(mismatches) == 0, f"Turn count mismatches: {mismatches}"

    def test_all_results_have_chief_complaint(self, mock_assistant, sample_scenarios):
        """All results must have a non-empty final_record.chief_complaint."""
        missing = []
        for scenario in sample_scenarios:
            result = mock_assistant.run_scenario(scenario)
            if not result.final_record.chief_complaint:
                missing.append(scenario["scenario_id"])
        assert len(missing) == 0, f"Results with empty chief_complaint: {missing}"

    def test_average_confidence_above_threshold(self, mock_assistant, sample_scenarios):
        """Mean overall_confidence across all 10 scenarios must exceed 0.5."""
        confidences = []
        for scenario in sample_scenarios:
            result = mock_assistant.run_scenario(scenario)
            confidences.append(result.final_record.overall_confidence)
        avg = sum(confidences) / len(confidences)
        print(f"\nSection A avg confidence: {avg:.3f}")
        assert avg > 0.5, f"Average confidence {avg:.3f} is not above 0.5"


# ===========================================================================
# SECTION B — Red flag category coverage
# ===========================================================================

RED_FLAG_SCENARIOS = {
    "cardiac": ["scenario_01"],
    "neurological_emergency": ["scenario_05"],
    "anaphylaxis": ["scenario_06"],  # category is "dermatology" but expected_red_flag=True
    "mental_health": ["scenario_07"],
    "obstetric_emergency": ["scenario_03"],
    "chronic_diabetes": ["scenario_08"],
}


class TestSectionB:
    """Verify that each critical red-flag category is reliably detected."""

    def test_cardiac_red_flag_detected(self, mock_assistant, sample_scenarios):
        """scenario_01 (cardiac) must produce non-empty red_flags."""
        scenario = _get_scenario(sample_scenarios, "scenario_01")
        result = mock_assistant.run_scenario(scenario)
        assert bool(result.final_record.red_flags), \
            f"Expected cardiac red flags in scenario_01, got: {result.final_record.red_flags}"

    def test_stroke_red_flag_detected(self, mock_assistant, sample_scenarios):
        """scenario_05 (neurological emergency / stroke) must produce non-empty red_flags."""
        scenario = _get_scenario(sample_scenarios, "scenario_05")
        result = mock_assistant.run_scenario(scenario)
        assert bool(result.final_record.red_flags), \
            f"Expected neurological red flags in scenario_05, got: {result.final_record.red_flags}"

    def test_anaphylaxis_red_flag_detected(self, mock_assistant, sample_scenarios):
        """scenario_06 (anaphylaxis) must produce non-empty red_flags."""
        scenario = _get_scenario(sample_scenarios, "scenario_06")
        result = mock_assistant.run_scenario(scenario)
        assert bool(result.final_record.red_flags), \
            f"Expected anaphylaxis red flags in scenario_06, got: {result.final_record.red_flags}"

    def test_suicidal_ideation_red_flag_detected(self, mock_assistant, sample_scenarios):
        """scenario_07 (mental health / suicidal ideation) must produce non-empty red_flags."""
        scenario = _get_scenario(sample_scenarios, "scenario_07")
        result = mock_assistant.run_scenario(scenario)
        assert bool(result.final_record.red_flags), \
            f"Expected mental health red flags in scenario_07, got: {result.final_record.red_flags}"

    def test_obstetric_emergency_red_flag_detected(self, mock_assistant, sample_scenarios):
        """scenario_03 (obstetric emergency) must produce non-empty red_flags."""
        scenario = _get_scenario(sample_scenarios, "scenario_03")
        result = mock_assistant.run_scenario(scenario)
        assert bool(result.final_record.red_flags), \
            f"Expected obstetric red flags in scenario_03, got: {result.final_record.red_flags}"

    def test_diabetic_foot_red_flag_detected(self, mock_assistant, sample_scenarios):
        """scenario_08 (chronic diabetes / diabetic foot) must produce non-empty red_flags."""
        scenario = _get_scenario(sample_scenarios, "scenario_08")
        result = mock_assistant.run_scenario(scenario)
        assert bool(result.final_record.red_flags), \
            f"Expected diabetic red flags in scenario_08, got: {result.final_record.red_flags}"

    def test_routine_scenarios_no_false_positives(self, mock_assistant, sample_scenarios):
        """Scenarios 04, 09, 10 have expected_red_flag=False — must produce zero red_flags."""
        routine_ids = ["scenario_04", "scenario_09", "scenario_10"]
        false_positives = []
        for sid in routine_ids:
            scenario = _get_scenario(sample_scenarios, sid)
            result = mock_assistant.run_scenario(scenario)
            if bool(result.final_record.red_flags):
                false_positives.append({
                    "scenario_id": sid,
                    "red_flags": result.final_record.red_flags,
                })
        assert len(false_positives) == 0, \
            f"False positive red flags in routine scenarios: {false_positives}"


# ===========================================================================
# SECTION C — Language coverage
# ===========================================================================

class TestSectionC:
    """All supported languages must produce passing results with extracted chief_complaint."""

    def test_english_scenarios_pass(self, mock_assistant, sample_scenarios):
        """All EN-language scenarios must pass D4 criterion."""
        en_scenarios = [s for s in sample_scenarios if s.get("language", "en").upper() == "EN"]
        assert len(en_scenarios) > 0, "No English scenarios found in SAMPLE_SCENARIOS"
        failed = []
        for scenario in en_scenarios:
            result = mock_assistant.run_scenario(scenario)
            if not result.passed_d4_criterion:
                failed.append(scenario["scenario_id"])
        assert len(failed) == 0, f"English scenarios failed D4: {failed}"

    def test_spanish_scenarios_pass(self, mock_assistant, sample_scenarios):
        """Scenarios 02 and 06 (ES) must pass D4 and produce non-empty chief_complaint."""
        es_scenarios = [s for s in sample_scenarios if s.get("language", "").upper() == "ES"]
        if not es_scenarios:
            pytest.skip("No Spanish (ES) scenarios found in SAMPLE_SCENARIOS")
        failed = []
        missing_complaint = []
        for scenario in es_scenarios:
            result = mock_assistant.run_scenario(scenario)
            if not result.passed_d4_criterion:
                failed.append(scenario["scenario_id"])
            if not result.final_record.chief_complaint:
                missing_complaint.append(scenario["scenario_id"])
        assert len(failed) == 0, f"Spanish scenarios failed D4: {failed}"
        assert len(missing_complaint) == 0, \
            f"Spanish scenarios missing chief_complaint: {missing_complaint}"

    def test_french_scenarios_pass(self, mock_assistant, sample_scenarios):
        """Scenarios 03 and 08 (FR) must pass D4 criterion."""
        fr_scenarios = [s for s in sample_scenarios if s.get("language", "").upper() == "FR"]
        if not fr_scenarios:
            pytest.skip("No French (FR) scenarios found in SAMPLE_SCENARIOS")
        failed = []
        for scenario in fr_scenarios:
            result = mock_assistant.run_scenario(scenario)
            if not result.passed_d4_criterion:
                failed.append(scenario["scenario_id"])
        assert len(failed) == 0, f"French scenarios failed D4: {failed}"

    def test_all_languages_produce_responses(self, mock_assistant, sample_scenarios):
        """Each language group must have at least 1 passing scenario."""
        by_lang = {}
        for scenario in sample_scenarios:
            lang = scenario.get("language", "en").upper()
            by_lang.setdefault(lang, []).append(scenario)

        failures = []
        for lang, group in by_lang.items():
            results = [mock_assistant.run_scenario(s) for s in group]
            passing = [r for r in results if r.passed_d4_criterion]
            if len(passing) == 0:
                failures.append(lang)
            print(f"  Lang {lang}: {len(passing)}/{len(results)} passed")

        assert len(failures) == 0, \
            f"Languages with zero passing scenarios: {failures}"


# ===========================================================================
# SECTION D — Extended corpus batch
# ===========================================================================

_MOCK_SKIP_REASON = (
    "Extended corpus tests require --use-real-model. "
    "The mock model only has scripted responses for the original 10 scenarios; "
    "synthetic gen_* IDs all return the default fallback (no chief_complaint extracted). "
    "These thresholds are only meaningful against the real Gemma 4 model."
)


@pytest.mark.integration
class TestSectionD:
    """Extended corpus tests (500 scenarios). Run with --max-scenarios to control batch size.
    NOTE: these tests are skipped in mock mode — they require --use-real-model."""

    def test_extended_corpus_pass_rate(self, mock_assistant, extended_scenarios, max_scenarios, using_real_model):
        """Overall pass rate on extended corpus must be >= 80%."""
        if not using_real_model:
            pytest.skip(_MOCK_SKIP_REASON)
        n = min(max_scenarios, len(extended_scenarios))
        pairs = _run_subset(mock_assistant, extended_scenarios, n)
        total = len(pairs)
        passed = sum(1 for _, r in pairs if r.passed_d4_criterion)
        pass_rate = passed / total if total > 0 else 0.0
        print(f"\nExtended corpus pass rate: {pass_rate:.1%} ({passed}/{total})")
        assert pass_rate >= 0.80, \
            f"Extended corpus pass rate {pass_rate:.1%} is below 0.80 threshold"

    def test_extended_corpus_red_flag_recall(self, mock_assistant, extended_scenarios, max_scenarios, using_real_model):
        """Red flag recall on expected-positive subset must be >= 70%."""
        if not using_real_model:
            pytest.skip(_MOCK_SKIP_REASON)
        n = min(max_scenarios, len(extended_scenarios))
        pairs = _run_subset(mock_assistant, extended_scenarios, n)

        expected_positive = [(s, r) for s, r in pairs if s.get("expected_red_flag", False)]
        if not expected_positive:
            pytest.skip("No red-flag-positive scenarios in extended corpus subset")

        actually_flagged = sum(
            1 for _, r in expected_positive if bool(r.final_record.red_flags)
        )
        recall = actually_flagged / len(expected_positive)
        print(f"\nExtended corpus red flag recall: {recall:.1%} "
              f"({actually_flagged}/{len(expected_positive)})")
        assert recall >= 0.70, \
            f"Red flag recall {recall:.1%} is below 0.70 threshold"

    def test_extended_corpus_red_flag_precision(self, mock_assistant, extended_scenarios, max_scenarios, using_real_model):
        """False positive rate on expected-negative subset must be <= 30%."""
        if not using_real_model:
            pytest.skip(_MOCK_SKIP_REASON)
        n = min(max_scenarios, len(extended_scenarios))
        pairs = _run_subset(mock_assistant, extended_scenarios, n)

        expected_negative = [(s, r) for s, r in pairs if not s.get("expected_red_flag", False)]
        if not expected_negative:
            pytest.skip("No red-flag-negative scenarios in extended corpus subset")

        flagged_when_shouldnt = sum(
            1 for _, r in expected_negative if bool(r.final_record.red_flags)
        )
        false_positive_rate = flagged_when_shouldnt / len(expected_negative)
        print(f"\nExtended corpus red flag false positive rate: {false_positive_rate:.1%} "
              f"({flagged_when_shouldnt}/{len(expected_negative)})")
        assert false_positive_rate <= 0.30, \
            f"Red flag false positive rate {false_positive_rate:.1%} exceeds 0.30 threshold"

    def test_extended_corpus_chief_complaint_coverage(self, mock_assistant, extended_scenarios, max_scenarios, using_real_model):
        """chief_complaint extraction rate on extended corpus must be >= 85%."""
        if not using_real_model:
            pytest.skip(_MOCK_SKIP_REASON)
        n = min(max_scenarios, len(extended_scenarios))
        pairs = _run_subset(mock_assistant, extended_scenarios, n)

        total = len(pairs)
        extracted = sum(
            1 for _, r in pairs if bool(r.final_record.chief_complaint)
        )
        coverage = extracted / total if total > 0 else 0.0
        print(f"\nExtended corpus chief_complaint coverage: {coverage:.1%} ({extracted}/{total})")
        assert coverage >= 0.85, \
            f"chief_complaint coverage {coverage:.1%} is below 0.85 threshold"

    def test_extended_corpus_confidence_distribution(self, mock_assistant, extended_scenarios, max_scenarios, using_real_model):
        """Mean confidence across extended corpus must exceed 0.50. Prints distribution stats."""
        if not using_real_model:
            pytest.skip(_MOCK_SKIP_REASON)
        n = min(max_scenarios, len(extended_scenarios))
        pairs = _run_subset(mock_assistant, extended_scenarios, n)

        confidences = [r.final_record.overall_confidence for _, r in pairs]
        conf_min = min(confidences)
        conf_max = max(confidences)
        conf_mean = statistics.mean(confidences)
        conf_median = statistics.median(confidences)
        print(
            f"\nExtended corpus confidence distribution:\n"
            f"  min={conf_min:.3f}  max={conf_max:.3f}  "
            f"mean={conf_mean:.3f}  median={conf_median:.3f}"
        )
        assert conf_mean > 0.50, \
            f"Mean confidence {conf_mean:.3f} is not above 0.50"


# ===========================================================================
# SECTION E — Edge case robustness
# ===========================================================================

@pytest.mark.integration
class TestSectionE:
    """Ensure the assistant handles unusual inputs without raising exceptions."""

    def _make_scenario(self, scenario_id, turns, description="edge case",
                       expected_red_flag=False, language="en", category="edge"):
        return {
            "scenario_id": scenario_id,
            "language": language,
            "description": description,
            "category": category,
            "turns": turns,
            "expected_fields": ["chief_complaint"],
            "expected_red_flag": expected_red_flag,
            "demographic": {"age": 30, "gender": "male"},
            "source": "synthetic",
        }

    def test_minimal_input_does_not_crash(self, mock_assistant):
        """A single-word patient turn must not raise and must return a result."""
        scenario = self._make_scenario(
            scenario_id="edge_minimal",
            turns=["pain"],
            description="minimal single-word input",
        )
        result = mock_assistant.run_scenario(scenario)
        assert result is not None, "run_scenario returned None for minimal input"

    def test_empty_input_does_not_crash(self, mock_assistant):
        """An empty string patient turn must not raise and must return a result."""
        scenario = self._make_scenario(
            scenario_id="edge_empty",
            turns=[""],
            description="empty string input",
        )
        result = mock_assistant.run_scenario(scenario)
        assert result is not None, "run_scenario returned None for empty input"

    def test_very_long_input_does_not_crash(self, mock_assistant):
        """A very long patient turn must not raise and must return a result."""
        long_turn = (
            "I have been experiencing this terrible pain in my chest for the past three days. "
            "It started on Monday morning when I woke up and felt this heaviness on my left side. "
            "The pain radiates down my left arm and sometimes up to my jaw. "
            "I have a history of high blood pressure and I take lisinopril 10mg daily. "
            "I also take atorvastatin 20mg for cholesterol. "
            "I had a stress test done last year and it was normal but my cardiologist said to "
            "watch out for these symptoms. The pain gets worse when I climb stairs or walk fast. "
            "I would rate it about a 7 out of 10 right now."
        )
        scenario = self._make_scenario(
            scenario_id="edge_long_input",
            turns=[long_turn],
            description="very long single patient turn",
            expected_red_flag=True,
        )
        result = mock_assistant.run_scenario(scenario)
        assert result is not None, "run_scenario returned None for very long input"

    def test_consecutive_scenarios_no_state_leakage(self, mock_assistant, sample_scenarios):
        """Running scenario_01 twice must yield identical scenario_id and chief_complaint."""
        scenario = _get_scenario(sample_scenarios, "scenario_01")
        result1 = mock_assistant.run_scenario(scenario)
        result2 = mock_assistant.run_scenario(scenario)
        assert result1.scenario_id == result2.scenario_id, \
            "scenario_id differed between consecutive runs of the same scenario"
        assert result1.final_record.chief_complaint == result2.final_record.chief_complaint, (
            f"chief_complaint differed between runs: "
            f"{result1.final_record.chief_complaint!r} vs {result2.final_record.chief_complaint!r}"
        )

    def test_different_scenarios_independent_records(self, mock_assistant, sample_scenarios):
        """Running scenario_01 then scenario_04 — no red-flag state must leak across runs."""
        scenario_01 = _get_scenario(sample_scenarios, "scenario_01")
        scenario_04 = _get_scenario(sample_scenarios, "scenario_04")
        # Run cardiac scenario first to exercise any shared state
        mock_assistant.run_scenario(scenario_01)
        result_04 = mock_assistant.run_scenario(scenario_04)
        assert not bool(result_04.final_record.red_flags), (
            f"scenario_04 should have no red flags, but got: {result_04.final_record.red_flags}. "
            "Possible state leakage from scenario_01."
        )


# ===========================================================================
# SECTION F — Metrics summary report
# ===========================================================================

@pytest.mark.integration
class TestSectionF:
    """Print a full metrics summary table. Never fails — purely informational."""

    def test_print_full_metrics_summary(
        self, mock_assistant, sample_scenarios, extended_scenarios, max_scenarios
    ):
        """Compute and print a formatted metrics table across original + extended scenarios."""
        # Run all 10 original scenarios
        original_pairs = [(s, mock_assistant.run_scenario(s)) for s in sample_scenarios]

        # Run up to 40 extended (bounded by max_scenarios and corpus size)
        extended_limit = min(40, max_scenarios, len(extended_scenarios))
        extended_pairs = _run_subset(mock_assistant, extended_scenarios, extended_limit)

        all_pairs = original_pairs + extended_pairs
        total = len(all_pairs)

        # Pass rate
        passed = sum(1 for _, r in all_pairs if r.passed_d4_criterion)
        pass_rate = passed / total if total > 0 else 0.0

        # Red flag recall
        pos_pairs = [(s, r) for s, r in all_pairs if s.get("expected_red_flag", False)]
        if pos_pairs:
            flagged = sum(1 for _, r in pos_pairs if bool(r.final_record.red_flags))
            recall = flagged / len(pos_pairs)
        else:
            recall = 0.0

        # Red flag false positive rate
        neg_pairs = [(s, r) for s, r in all_pairs if not s.get("expected_red_flag", False)]
        if neg_pairs:
            fp = sum(1 for _, r in neg_pairs if bool(r.final_record.red_flags))
            fp_rate = fp / len(neg_pairs)
        else:
            fp_rate = 0.0

        # Chief complaint coverage
        extracted = sum(1 for _, r in all_pairs if bool(r.final_record.chief_complaint))
        cc_coverage = extracted / total if total > 0 else 0.0

        # Average confidence
        confidences = [r.final_record.overall_confidence for _, r in all_pairs]
        avg_conf = statistics.mean(confidences) if confidences else 0.0

        # Language breakdown
        lang_counts = {}
        for s, _ in all_pairs:
            lang = s.get("language", "en").upper()
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        en_count = lang_counts.get("EN", 0)
        es_count = lang_counts.get("ES", 0)
        fr_count = lang_counts.get("FR", 0)

        # Print formatted table
        print(
            f"\n"
            f"╔══════════════════════════════════════════════════╗\n"
            f"║         CAREVOICE STRESS-TEST METRICS            ║\n"
            f"╠══════════════════════════════════════════════════╣\n"
            f"║ Total scenarios run:       {total:<22}║\n"
            f"║ Pass rate:                 {f'{pass_rate:.1%}':<22}║\n"
            f"║ Red flag recall:           {f'{recall:.1%}':<22}║\n"
            f"║ Red flag false pos rate:   {f'{fp_rate:.1%}':<22}║\n"
            f"║ Chief complaint coverage:  {f'{cc_coverage:.1%}':<22}║\n"
            f"║ Avg confidence:            {avg_conf:<22.3f}║\n"
            f"║ Language EN / ES / FR:     {en_count} / {es_count} / {fr_count:<17}║\n"
            f"╚══════════════════════════════════════════════════╝"
        )
        assert True
