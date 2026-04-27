#!/usr/bin/env python3
"""
Convert a Jupytext percent-format .py to a .ipynb
Handles:
  # %% [markdown]
  \"\"\"...markdown...\"\"\"
  # %%
  <code>
"""
import json, re, sys, pathlib, textwrap

def py_to_ipynb(src: str) -> dict:
    CELL_SEP   = re.compile(r"^# %%(.*)$", re.MULTILINE)
    MD_TRIPLE  = re.compile(r'^\s*"""(.*?)"""\s*$', re.DOTALL)

    cells = []
    # Split on cell separator markers
    parts = CELL_SEP.split(src)

    # parts[0] is any preamble before first %%, which we ignore
    # then alternating: tag, content, tag, content ...
    i = 1
    while i < len(parts):
        tag     = parts[i].strip()       # e.g. "" or "[markdown]"
        content = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        is_md = "[markdown]" in tag

        cell_id = f"{len(cells):02x}" + "a" * 30  # deterministic 32-char id

        if is_md:
            # Strip the surrounding triple-quote docstring if present
            m = MD_TRIPLE.match(content.strip())
            src_text = m.group(1) if m else content
            # Dedent
            src_text = textwrap.dedent(src_text).strip()
            cells.append({
                "id": cell_id,
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + "\n" for l in src_text.splitlines()],
            })
        else:
            code = content.strip("\n")
            if not code:
                continue
            cells.append({
                "id": cell_id,
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in code.splitlines()],
            })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,   # 4.5+ requires cell ids
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [
                    {
                        "modelInstanceType": "Gemma4E4bIt",
                        "sourceType": "model",
                        "modelId": "google/gemma-4/transformers/gemma-4-e4b-it/1"
                    }
                ],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
            },
        },
        "cells": cells,
    }


if __name__ == "__main__":
    src_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
               pathlib.Path(__file__).parent / "carevoice_trimodal_notebook.py"
    dst_path = src_path.with_suffix(".ipynb")

    nb = py_to_ipynb(src_path.read_text(encoding="utf-8"))
    dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md   = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"✅ Written: {dst_path}")
    print(f"   {n_md} markdown cells + {n_code} code cells = {len(nb['cells'])} total")
