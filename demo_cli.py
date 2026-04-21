"""CareVoice milestone validation CLI.

Runs scenario batches and prints structured output.
No internet calls during inference. Use --mock to skip model loading.

Usage:
    python demo_cli.py --mock                            # D4: 3 scenarios, local dev
    python demo_cli.py --mock --milestone d12            # D12: all 10 scenarios
    python demo_cli.py --model google/gemma-4-7b-it      # real Gemma 4 on GPU
    python demo_cli.py --model /kaggle/input/gemma-4     # Kaggle model path
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure UTF-8 output on Windows terminals that default to GBK/CP1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running from solutions/gemma_hackathon/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solutions.gemma_hackathon.intake_assistant import IntakeAssistant
from solutions.gemma_hackathon.scenarios import D4_SCENARIOS, D12_SCENARIOS


def run_milestone(
    assistant: IntakeAssistant,
    scenarios: list[dict],
    milestone: str,
    criterion: str,
    output_path: Path | None = None,
) -> dict:
    results = []
    all_passed = True

    print("=" * 60)
    print(f"CareVoice — {milestone} Milestone Validation")
    print("Offline Clinical Intake Assistant (Gemma 4 Good Hackathon)")
    print("=" * 60)

    for scenario in scenarios:
        print(f"\n[{scenario['scenario_id']}] {scenario['description']}")
        print("-" * 50)

        result = assistant.run_scenario(scenario)

        for turn in result.turns:
            print(f"\n  Patient: {turn.patient_input}")
            print(f"  CareVoice: {turn.assistant_message}")
            if turn.red_flag:
                print(f"  [!] RED FLAG: {turn.red_flag_reason}")
            print(f"  [extracted: {turn.extracted_field}={turn.extracted_value!r} conf={turn.confidence:.2f}]")

        print(f"\n  Final Record:")
        print("  " + result.final_record.to_provider_summary().replace("\n", "\n  "))
        print(f"\n  PASS: {result.passed_d4_criterion}")
        for note in result.notes:
            print(f"  {note}")

        results.append(asdict(result))
        if not result.passed_d4_criterion:
            all_passed = False

    # D12+ adds rubric scoring
    rubric_dict = None
    accessibility_dict = None
    if milestone.upper() in ("D12", "D20"):
        from solutions.gemma_hackathon.self_eval import score_d12_results, print_rubric
        rubric = score_d12_results(results)
        print_rubric(rubric)
        rubric_dict = rubric.to_dict()

    # D20 adds accessibility check
    accessibility_dict = None
    if milestone.upper() == "D20":
        from solutions.gemma_hackathon.accessibility_check import run_d20_validation
        acc = run_d20_validation()
        accessibility_dict = {
            "fits_8gb_4bit": acc.fits_8gb_4bit,
            "all_languages_pass": acc.all_languages_pass,
            "passed": acc.passed,
        }

    summary = {
        "milestone": milestone.upper(),
        "criterion": criterion,
        "scenarios_run": len(results),
        "all_passed": all_passed,
        "results": results,
    }
    if rubric_dict:
        summary["rubric"] = rubric_dict
    if accessibility_dict:
        summary["accessibility"] = accessibility_dict
        if not accessibility_dict["passed"]:
            all_passed = False

    print("\n" + "=" * 60)
    print(f"{milestone.upper()} MILESTONE: {'PASS' if all_passed else 'FAIL'}")
    print(f"Scenarios passed: {sum(1 for r in results if r['passed_d4_criterion'])}/{len(results)}")
    print("=" * 60)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nOutput written to: {output_path}")

    return summary


MILESTONE_CONFIG = {
    "d4": {
        "scenarios": D4_SCENARIOS,
        "criterion": "Runs locally, produces structured output, no internet calls",
    },
    "d12": {
        "scenarios": D12_SCENARIOS,
        "criterion": "10 sample patient scenarios produce coherent summaries",
    },
    "d20": {
        "scenarios": D12_SCENARIOS,
        "criterion": "Runs on 8GB RAM laptop, end-to-end flow works in all 3 languages",
    },
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CareVoice milestone validation")
    p.add_argument("--mock", action="store_true", help="Use mock model (no GPU)")
    p.add_argument("--milestone", default="d4", choices=list(MILESTONE_CONFIG),
                   help="Which milestone to run (default: d4)")
    p.add_argument("--model", default="google/gemma-4-7b-it", help="Model ID or path")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    p.add_argument("--output", help="Write JSON result to this path")
    args = p.parse_args(argv)

    if args.mock:
        print(f"Using mock model (local validation mode, milestone={args.milestone})")
        assistant = IntakeAssistant.mock()
    else:
        assistant = IntakeAssistant.load(
            model_id=args.model,
            use_4bit=not args.no_4bit,
        )

    cfg = MILESTONE_CONFIG[args.milestone]
    output_path = Path(args.output) if args.output else None
    summary = run_milestone(
        assistant,
        scenarios=cfg["scenarios"],
        milestone=args.milestone,
        criterion=cfg["criterion"],
        output_path=output_path,
    )

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
