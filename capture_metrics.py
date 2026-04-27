#!/usr/bin/env python3
"""
Pull carevoice_results.json from completed Kaggle kernel
and update submission_form.md with real metrics.

Usage: python capture_metrics.py
"""
import subprocess, json, pathlib, sys, re, datetime

KERNEL_SLUG = "farmountain/carevoice-gemma4-clinical-intake"
OUT_DIR     = pathlib.Path("/tmp/kout_final")
FORM_PATH   = pathlib.Path(__file__).parent / "submission_form.md"

def pull_output():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(OUT_DIR)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return False
    return True

def load_results():
    candidates = list(OUT_DIR.glob("*.json"))
    # prefer carevoice_results.json
    for p in candidates:
        if "carevoice" in p.name.lower():
            return json.loads(p.read_text())
    if candidates:
        return json.loads(candidates[0].read_text())
    return None

def patch_form(results: dict):
    form = FORM_PATH.read_text(encoding="utf-8")

    s1  = results.get("scene_1_pass", False)
    s2a = results.get("scene_2_image_accuracy", 0)
    s2n = results.get("scene_2_n_samples", 0)
    s3  = results.get("scene_3_audio_samples", 0)
    s4  = results.get("scene_4_multilingual", False)
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    scene_block = f"""```
Scene 1  Red flag + multilingual (3 languages)   {'PASS' if s1 else 'FAIL'}
Scene 2  Image triage accuracy                   {s2a*100:.0f}% ({s2n} samples)
Scene 3  Audio respiratory analysis              {s3} recordings processed
Scene 4  Multilingual                            {'Auto-detected 3 languages ✅' if s4 else 'PARTIAL'}
```

*Metrics captured from Kaggle kernel v17 output on {ts}*"""

    # Replace the placeholder scene summary block
    form = re.sub(
        r"```\nScene 1.*?```(\n\*Metrics captured.*?\*)?\n",
        scene_block + "\n",
        form, flags=re.DOTALL
    )

    FORM_PATH.write_text(form, encoding="utf-8")
    print(f"✅ submission_form.md updated with real metrics")
    print(f"   Scene 1: {'PASS' if s1 else 'FAIL'}")
    print(f"   Scene 2: {s2a*100:.0f}% image triage accuracy")
    print(f"   Scene 3: {s3} audio recordings")
    print(f"   Scene 4: {'✅' if s4 else 'partial'}")

if __name__ == "__main__":
    print(f"Pulling output from {KERNEL_SLUG}…")
    if not pull_output():
        sys.exit(1)
    results = load_results()
    if not results:
        print("No carevoice_results.json found in output — kernel may not have finished.")
        print("Files in output dir:", list(OUT_DIR.iterdir()))
        sys.exit(1)
    print("Results:", json.dumps(results, indent=2, default=str)[:800])
    patch_form(results)
