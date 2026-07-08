"""Qwen3 Megatron true-on-policy match test runner.

Wraps scripts/run_qwen3_4b.py with logprob/hidden-state capture enabled.
Runs in debug_one_sample mode (fast, 1 GPU).

Edit the CONFIG section below, then:
    python tools/true-on-policy/run_megatron.py [--skip-prepare]

After the run:
    python tools/true-on-policy/compute_metrics.py --save-dir <SAVE_DIR>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these paths before running
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen3-0.6B"      # "Qwen3-0.6B" (1 GPU, TP=1) or "Qwen3-4B" (needs more GPUs)
MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"
OUTPUT_DIR = "/root/output"
MEGATRON_PATH = "/root/Megatron-LM"
SAVE_DIR = "/tmp/true-on-policy"
CAPTURE_HIDDEN_STATES = True    # set False to skip megatron_hs_hook (saves memory)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]
_HOOK_PATH = f"{_REPO_ROOT}/tools/true-on-policy/megatron_hs_hook.py:register"


def _build_args():
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.run_qwen3_4b import ScriptArgs

    extra = ""
    if CAPTURE_HIDDEN_STATES:
        extra += f"--custom-megatron-before-log-prob-hook-path {_HOOK_PATH} "

    return ScriptArgs(
        mode="match",
        model_name=MODEL_NAME,
        train_backend="megatron",
        true_on_policy=True,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        megatron_path=MEGATRON_PATH,
        extra_args=extra.strip(),
        use_kl_loss=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true", help="Skip model download and checkpoint conversion")
    cli = parser.parse_args()

    # Must be set before execute() spawns subprocesses so workers inherit it.
    os.environ["MILES_TRUE_ON_POLICY_SAVE_DIR"] = SAVE_DIR

    from scripts.run_qwen3_4b import execute, prepare

    args = _build_args()

    if not cli.skip_prepare:
        prepare(args)

    execute(args)

    print(f"\nDone. Compute metrics:")
    print(f"  python tools/true-on-policy/compute_metrics.py --save-dir {SAVE_DIR}")


if __name__ == "__main__":
    main()
