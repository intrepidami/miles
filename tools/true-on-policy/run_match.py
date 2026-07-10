"""Qwen3 Megatron true-on-policy match test runner.

Wraps scripts/run_qwen3_4b.py with logprob/hidden-state capture enabled.
Output is tee'd to log/train_YYYYMMDD_HHMMSS.txt automatically.

Edit the CONFIG section below, then:
    python tools/true-on-policy/run_match.py [--skip-prepare]
    python tools/true-on-policy/run_match.py --cuda-visible-devices 0,1 --num-gpus-per-node 2 --num-nodes 1

After the run:
    python tools/true-on-policy/compute_metrics.py --save-dir <SAVE_DIR>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these paths before running
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen3-0.6B"  # "Qwen3-0.6B" (1 GPU, TP=1) or "Qwen3-4B" (needs more GPUs)
MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"
OUTPUT_DIR = "/root/output"
MEGATRON_PATH = "/root/Megatron-LM"
BASE_DIR = "/root/true-on-policy"  # each run saves to BASE_DIR/{YYYYMMDD_HHMMSS}/
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]
_HOOK_PATH = f"{_REPO_ROOT}/tools/true-on-policy/megatron_hs_hook.py:register"
_LOG_GUARD = "_MILES_TRUE_ON_POLICY_LOGGED"
_RUN_TIMESTAMP_VAR = "_MILES_RUN_TIMESTAMP"


def _build_args(
    dumper_dir: Path,
    train_backend: str,
    true_on_policy: bool,
    capture_hidden_states: bool,
    dumper_enable: bool,
    num_gpus_per_node: int | None = None,
    num_nodes: int | None = None,
    megatron_tp: int | None = None,
    megatron_pp: int | None = None,
    megatron_cp: int | None = None,
    megatron_dp: int | None = None,
):
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.run_qwen3_4b import ScriptArgs

    extra = ""
    if capture_hidden_states:
        if train_backend != "megatron":
            print("WARNING: --capture-hidden-states uses a Megatron-only hook; ignored for fsdp backend")
        else:
            extra += f"--custom-megatron-before-log-prob-hook-path {_HOOK_PATH} "
    if dumper_enable:
        _dumper_filter = 'layer_id is not None and name is not None and name.endswith(".mlp.output")'
        extra += (
            f"--dumper-enable "
            f"--dumper-dir {dumper_dir} "
            f"--dumper-fwd-only enable=true non_intrusive_mode=all filter='{_dumper_filter}' "
            f"--dumper-inference enable=true non_intrusive_mode=all filter='{_dumper_filter}' "
        )

    kwargs: dict = dict(
        mode="match",
        model_name=MODEL_NAME,
        train_backend=train_backend,
        true_on_policy=true_on_policy,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        megatron_path=MEGATRON_PATH,
        extra_args=extra.strip(),
        use_kl_loss=False,
    )
    if num_gpus_per_node is not None:
        kwargs["num_gpus_per_node"] = num_gpus_per_node
    if num_nodes is not None:
        kwargs["num_nodes"] = num_nodes
    if any(v is not None for v in (megatron_tp, megatron_pp, megatron_cp, megatron_dp)) and train_backend != "megatron":
        print("WARNING: --megatron-tp/--megatron-pp/--megatron-cp/--megatron-dp only apply to the megatron backend; ignored")
    if megatron_tp is not None:
        kwargs["megatron_tp_size"] = megatron_tp
    if megatron_pp is not None:
        kwargs["megatron_pp_size"] = megatron_pp
    if megatron_cp is not None:
        kwargs["megatron_cp_size"] = megatron_cp
    if megatron_dp is not None:
        kwargs["megatron_dp_size"] = megatron_dp
    return ScriptArgs(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true", help="Skip model download and checkpoint conversion")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        metavar="IDS",
        help="Value for CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1,2,3')",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=None,
        help="Override num_gpus_per_node (default: derived from hardware)",
    )
    parser.add_argument(
        "--num-nodes", type=int, default=None, help="Override num_nodes (default: 1 or SLURM_JOB_NUM_NODES)"
    )
    parser.add_argument(
        "--train-backend",
        choices=["megatron", "fsdp"],
        default="megatron",
        help="Training backend (default: megatron)",
    )
    parser.add_argument("--true-on-policy", action="store_true", help="Enable true-on-policy mode")
    parser.add_argument(
        "--megatron-tp", type=int, default=None, help="Megatron tensor-model-parallel size (default: model-derived)"
    )
    parser.add_argument(
        "--megatron-pp", type=int, default=None, help="Megatron pipeline-model-parallel size (default: 1)"
    )
    parser.add_argument(
        "--megatron-cp", type=int, default=None, help="Megatron context-parallel size (default: model-derived)"
    )
    parser.add_argument(
        "--megatron-dp", type=int, default=None, help="Megatron data-parallel size (default: 1)"
    )
    parser.add_argument(
        "--capture-hidden-states", action="store_true", help="Enable Megatron hidden-state capture hook"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (default: disabled)")
    parser.add_argument("--dumper-enable", action="store_true", help="Enable SGLang/Megatron tensor dumper")
    cli = parser.parse_args()

    if cli.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli.cuda_visible_devices

    if not cli.wandb:
        # get_default_wandb_args() enables wandb only when WANDB_API_KEY is set;
        # match runs are judged offline via compute_metrics.py, so default off.
        os.environ.pop("WANDB_API_KEY", None)
        os.environ.setdefault("WANDB_MODE", "disabled")

    # Tee stdout+stderr to a log file inside the timestamped run directory.
    if not os.environ.get(_LOG_GUARD):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(BASE_DIR) / run_timestamp
        log_dir = save_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"log_{run_timestamp}.txt"
        print(f"Logging to {log_path}", flush=True)
        env = os.environ.copy()
        env[_LOG_GUARD] = "1"
        env[_RUN_TIMESTAMP_VAR] = run_timestamp
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(
                [sys.executable] + sys.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                lf.write(line)
        sys.exit(proc.wait())

    # Timestamped run directory: BASE_DIR/{YYYYMMDD_HHMMSS}/
    run_timestamp = os.environ.get(_RUN_TIMESTAMP_VAR) or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(BASE_DIR) / run_timestamp
    dumper_dir = save_dir / "tensor_cmp"

    # Must be set before execute() spawns subprocesses so workers inherit it.
    os.environ["MILES_TRUE_ON_POLICY_SAVE_DIR"] = str(save_dir)

    from scripts.run_qwen3_4b import execute, prepare

    args = _build_args(
        dumper_dir=dumper_dir,
        train_backend=cli.train_backend,
        true_on_policy=cli.true_on_policy,
        capture_hidden_states=cli.capture_hidden_states,
        dumper_enable=cli.dumper_enable,
        num_gpus_per_node=cli.num_gpus_per_node,
        num_nodes=cli.num_nodes,
        megatron_tp=cli.megatron_tp,
        megatron_pp=cli.megatron_pp,
        megatron_cp=cli.megatron_cp,
        megatron_dp=cli.megatron_dp,
    )

    if not cli.skip_prepare:
        prepare(args)

    execute(args)

    # Move dump_details from OUTPUT_DIR/{run_id}/dump_details/ into the timestamped save_dir.
    # Use this run's run_id, not the newest-mtime dir: the latter can grab a
    # previous run's data when the current run produced none.
    dump_details_src = Path(OUTPUT_DIR) / args.run_id / "dump_details"
    if dump_details_src.exists():
        import shutil

        dump_details_dst = save_dir / "dump_details"
        shutil.move(str(dump_details_src), str(dump_details_dst))
        print(f"Moved dump_details → {dump_details_dst}")

    print(f"\nDone. Compute metrics:")
    print(f"  python tools/true-on-policy/compute_metrics.py --save-dir {save_dir}")


if __name__ == "__main__":
    main()
