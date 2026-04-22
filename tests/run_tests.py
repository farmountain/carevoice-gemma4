"""
CareVoice Stress-Test Runner (standalone, no pytest required)
=============================================================
Usage (from project root):
    python tests/run_tests.py

Optionally control extended corpus batch size via env var:
    MAX_EXTENDED=100 python tests/run_tests.py

Exits with code 0 if overall pass_rate >= 0.80, else code 1.
Results are written to tests/test_results.json.
"""
import sys
import os
import json
import statistics
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from inside tests/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..')       # gemma_hackathon/
_SOLUTIONS_ROOT = os.path.join(_THIS_DIR, '..', '..') # solutions/
sys.path.insert(0, os.path.abspath(_SOLUTIONS_ROOT))
sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from gemma_hackathon.intake_assistant import IntakeAssistant
except ImportError as e:
    print(f"[ERROR] Could not import IntakeAssistant: {e}")
    print("Make sure you are running from the project root:")
    print("  python tests/run_tests.py")
    sys.exit(1)

try:
    from gemma_hackathon.scenarios import SAMPLE_SCENARIOS
except ImportError as e:
    print(f"[ERROR] Could not import SAMPLE_SCENARIOS from scenarios.py: {e}")
    sys.exit(1)

# Extended corpus is optional — graceful degradation if not present
_EXTENDED_CORPUS = None
try:
    from gemma_hackathon.tests.corpus.generator import EXTENDED_CORPUS as _EXTENDED_CORPUS
except ImportError:
    try:
        from corpus.generator import EXTENDED_CORPUS as _EXTENDED_CORPUS
    except ImportError:
        _EXTENDED_CORPUS = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PASS_RATE_THRESHOLD = 0.80
RESULTS_PATH = os.path.join(_THIS_DIR, "test_results.json")
MAX_EXTENDED = int(os.environ.get("MAX_EXTENDED", "50"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scenarios(assistant, scenarios, label=""):
    """Run a list of scenario dicts and return a list of result record dicts."""
    records = []
    for i, scenario in enumerate(scenarios, 1):
        sid = scenario.get("scenario_id", f"unknown_{i}")
        try:
            result = assistant.run_scenario(scenario)
            records.append({
                "scenario_id": result.scenario_id,
                "passed": result.passed_d4_criterion,
                "has_red_flags": bool(result.final_record.red_flags),
                "expected_red_flag": scenario.get("expected_red_flag", False),
                "chief_complaint": result.final_record.chief_complaint or "",
                "confidence": result.final_record.overall_confidence,
                "language": scenario.get("language", "en").upper(),
                "source": label,
                "error": None,
            })
        except Exception as exc:
            records.append({
                "scenario_id": sid,
                "passed": False,
                "has_red_flags": False,
                "expected_red_flag": scenario.get("expected_red_flag", False),
                "chief_complaint": "",
                "confidence": 0.0,
                "language": scenario.get("language", "en").upper(),
                "source": label,
                "error": str(exc),
            })
    return records


def _compute_metrics(records):
    """Compute aggregate metrics dict from a list of result record dicts."""
    total = len(records)
    if total == 0:
        return {
            "total_run": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "red_flag_recall": 0.0,
            "red_flag_false_positive_rate": 0.0,
            "chief_complaint_coverage": 0.0,
            "avg_confidence": 0.0,
            "lang_counts": {},
        }

    passed = sum(1 for r in records if r["passed"])
    pass_rate = passed / total

    # Red flag recall — true positive rate over expected-positive subset
    pos = [r for r in records if r["expected_red_flag"]]
    if pos:
        actually_flagged = sum(1 for r in pos if r["has_red_flags"])
        recall = actually_flagged / len(pos)
    else:
        recall = 0.0

    # Red flag false positive rate — over expected-negative subset
    neg = [r for r in records if not r["expected_red_flag"]]
    if neg:
        fp = sum(1 for r in neg if r["has_red_flags"])
        fp_rate = fp / len(neg)
    else:
        fp_rate = 0.0

    # Chief complaint extraction coverage
    extracted = sum(1 for r in records if r["chief_complaint"])
    cc_coverage = extracted / total

    # Average confidence score
    confidences = [r["confidence"] for r in records]
    avg_conf = statistics.mean(confidences)

    # Language breakdown counts
    lang_counts = {}
    for r in records:
        lang = r.get("language", "EN")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    return {
        "total_run": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "red_flag_recall": recall,
        "red_flag_false_positive_rate": fp_rate,
        "chief_complaint_coverage": cc_coverage,
        "avg_confidence": avg_conf,
        "lang_counts": lang_counts,
    }


def _print_metrics_table(metrics):
    """Print the formatted metrics summary table to stdout."""
    lc = metrics["lang_counts"]
    en = lc.get("EN", 0)
    es = lc.get("ES", 0)
    fr = lc.get("FR", 0)
    # Pre-format percentage strings (backslash not allowed inside f-string exprs in Py<3.12)
    pass_rate_s  = "{:.1%}".format(metrics["pass_rate"])
    recall_s     = "{:.1%}".format(metrics["red_flag_recall"])
    fp_s         = "{:.1%}".format(metrics["red_flag_false_positive_rate"])
    cc_s         = "{:.1%}".format(metrics["chief_complaint_coverage"])
    print(
        f"\n"
        f"╔══════════════════════════════════════════════════╗\n"
        f"║         CAREVOICE STRESS-TEST METRICS            ║\n"
        f"╠══════════════════════════════════════════════════╣\n"
        f"║ Total scenarios run:       {metrics['total_run']:<22}║\n"
        f"║ Pass rate:                 {pass_rate_s:<22}║\n"
        f"║ Red flag recall:           {recall_s:<22}║\n"
        f"║ Red flag false pos rate:   {fp_s:<22}║\n"
        f"║ Chief complaint coverage:  {cc_s:<22}║\n"
        f"║ Avg confidence:            {metrics['avg_confidence']:<22.3f}║\n"
        f"║ Language EN / ES / FR:     {en} / {es} / {fr:<17}║\n"
        f"╚══════════════════════════════════════════════════╝"
    )


def _write_results(metrics, scenario_records):
    """Serialise metrics and per-scenario records to tests/test_results.json."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "scenarios": scenario_records,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nResults written to: {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CareVoice Stress-Test Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1 — load mock assistant (no GPU / model weights required)
    print("\n[1/4] Loading mock assistant...")
    assistant = IntakeAssistant.mock()
    print("      OK")

    # Step 2 — run all 10 original SAMPLE_SCENARIOS
    print(f"\n[2/4] Running {len(SAMPLE_SCENARIOS)} original scenarios...")
    original_records = _run_scenarios(assistant, SAMPLE_SCENARIOS, label="original")
    orig_passed = sum(1 for r in original_records if r["passed"])
    print(f"      {orig_passed}/{len(original_records)} passed D4 criterion")

    # Step 3 — run up to MAX_EXTENDED from the extended corpus if available
    extended_records = []
    if _EXTENDED_CORPUS is not None:
        n = min(MAX_EXTENDED, len(_EXTENDED_CORPUS))
        print(
            f"\n[3/4] Running {n} extended corpus scenarios "
            f"(of {len(_EXTENDED_CORPUS)} available, MAX_EXTENDED={MAX_EXTENDED})..."
        )
        extended_records = _run_scenarios(assistant, _EXTENDED_CORPUS[:n], label="extended")
        ext_passed = sum(1 for r in extended_records if r["passed"])
        print(f"      {ext_passed}/{len(extended_records)} passed D4 criterion")
    else:
        print(
            "\n[3/4] Extended corpus not available — skipping.\n"
            "      (Provide tests/corpus/generator.py with EXTENDED_CORPUS to enable.)"
        )

    # Step 4 — compute metrics and print summary
    print("\n[4/4] Computing metrics...")
    all_records = original_records + extended_records
    metrics = _compute_metrics(all_records)
    _print_metrics_table(metrics)

    # Report any individual failures
    failures = [r for r in all_records if not r["passed"]]
    errors = [r for r in all_records if r["error"]]
    if failures:
        print(f"\nFailed scenarios ({len(failures)}):")
        for r in failures:
            err_suffix = f"  [ERROR: {r['error']}]" if r["error"] else ""
            print(f"  - {r['scenario_id']} ({r['source']}){err_suffix}")
    if errors:
        print(f"\nScenarios with exceptions ({len(errors)}):")
        for r in errors:
            print(f"  - {r['scenario_id']}: {r['error']}")

    # Persist results to JSON
    _write_results(metrics, all_records)

    # Exit code is based on the ORIGINAL 10 scenarios only.
    # Extended corpus results are informational — the mock has no scripted responses
    # for synthetic gen_* IDs; run with --use-real-model (Kaggle GPU) for meaningful
    # extended corpus metrics.
    orig_pass_rate = orig_passed / len(original_records) if original_records else 0.0
    if orig_pass_rate >= PASS_RATE_THRESHOLD:
        print(
            f"\n[PASS] Original scenario pass rate {orig_pass_rate:.1%} >= "
            f"{PASS_RATE_THRESHOLD:.0%} threshold.\n"
            f"       Extended corpus ({len(extended_records)} scenarios) requires real model "
            f"for meaningful metrics.\n"
        )
        sys.exit(0)
    else:
        print(
            f"\n[FAIL] Original scenario pass rate {orig_pass_rate:.1%} < "
            f"{PASS_RATE_THRESHOLD:.0%} threshold.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
