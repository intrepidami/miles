"""Compute true-on-policy match metrics from saved logprobs and hidden states.

Two data sources are supported (mutually exclusive for logprob input):

  Mode 1 — .npy files (default):
    Reads rollout_*/log_probs.npy and rollout_*/rollout_log_probs.npy from --save-dir.
    Produced by MILES_TRUE_ON_POLICY_SAVE_DIR mechanism in log_utils.py.

  Mode 2 — dump_details .pt files (--dump-details):
    Reads dump_details/train_data/{rollout_id}_{rank}.pt, extracts log_probs and
    rollout_log_probs from the RolloutBatch stored there.
    Produced when dump_details is enabled in the training run.

CSV output is always written to <save_dir>/metrics/match.csv regardless of mode.

Usage:
    # Mode 1
    python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022
    python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022 --rollout 0

    # Mode 2
    python tools/true-on-policy/compute_metrics.py \\
        --save-dir /root/true-on-policy/20260708_143022 \\
        --dump-details /root/output/<run_id>/dump_details
    python tools/true-on-policy/compute_metrics.py \\
        --save-dir /root/true-on-policy/20260708_143022 \\
        --dump-details /root/output/<run_id>/dump_details --rank 0 --rollout 0 1
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _load_logprobs_from_dump_details(
    dump_details_dir: Path, rank: int, rollout_ids: list[int] | None
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    train_data_dir = dump_details_dir / "train_data"
    if rollout_ids is not None:
        files = sorted(train_data_dir / f"{r}_{rank}.pt" for r in rollout_ids)
    else:
        files = sorted(train_data_dir.glob(f"*_{rank}.pt"))

    if not files:
        raise FileNotFoundError(f"No train_data files for rank={rank} in {train_data_dir}")

    lp_parts, rlp_parts = [], []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        rb = data["rollout_data"]
        for key, parts in [("log_probs", lp_parts), ("rollout_log_probs", rlp_parts)]:
            val = rb.get(key) if hasattr(rb, "get") else getattr(rb, key, None)
            if val is None:
                print(f"  skip {f.name}: missing {key}")
                break
            if isinstance(val, (list, tuple)) and val:
                parts.append(torch.cat(val).float().numpy())
            else:
                parts.append(val.float().numpy())

    if not lp_parts:
        raise FileNotFoundError(f"No valid train_data entries in {train_data_dir}")
    return np.concatenate(lp_parts), np.concatenate(rlp_parts)


def _load_logprobs(save_dir: Path, rollout_ids: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    rollout_dirs = sorted(save_dir.glob("rollout_*"))
    if rollout_ids is not None:
        rollout_dirs = [save_dir / f"rollout_{r:04d}" for r in rollout_ids]

    lp_parts, rlp_parts = [], []
    for rd in rollout_dirs:
        lp_path = rd / "log_probs.npy"
        rlp_path = rd / "rollout_log_probs.npy"
        if not lp_path.exists() or not rlp_path.exists():
            print(f"  skip {rd.name}: missing log_probs.npy or rollout_log_probs.npy")
            continue
        lp_parts.append(np.load(lp_path))
        rlp_parts.append(np.load(rlp_path))

    if not lp_parts:
        raise FileNotFoundError(f"No valid rollout dirs found in {save_dir}")

    return np.concatenate(lp_parts), np.concatenate(rlp_parts)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
    return float(np.dot(a_c, b_c) / denom) if denom > 0 else float("nan")


def _load_megatron_hs(save_dir: Path) -> dict[int, np.ndarray] | None:
    """Load per-layer hidden states from rank_0 (PP=1 assumed)."""
    hs_dir = save_dir / "megatron_hs" / "rank_0"
    if not hs_dir.exists():
        return None
    layers = {}
    for f in sorted(hs_dir.glob("layer_*.npy")):
        idx = int(f.stem.split("_")[1])
        layers[idx] = np.load(f)  # [total_tokens, hidden_dim]
    return layers if layers else None


def _compute_hs_metrics(layers: dict[int, np.ndarray]) -> None:
    """Print per-layer L2 norm stats and cumulative MSE (self-consistency check)."""
    print("\nMegatron hidden-state per-layer L2 norm (mean across tokens):")
    norms = []
    for idx in sorted(layers):
        arr = layers[idx].astype(np.float64)
        norm = float(np.linalg.norm(arr, axis=-1).mean())
        norms.append((idx, norm))
        print(f"  layer {idx:3d}: mean_L2={norm:.4f}  shape={arr.shape}")

    cumulative_mse = float(np.mean([(n**2) for _, n in norms]))
    print(f"\nCumulative MSE (mean of squared L2 norms): {cumulative_mse:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute true-on-policy logprob match metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-dir", type=Path, required=True,
        help="Run directory (BASE_DIR/YYYYMMDD_HHMMSS/). Reads .npy files (Mode 1) and always writes metrics/match.csv.",
    )
    parser.add_argument(
        "--dump-details", type=Path, default=None, metavar="PATH",
        help="(Mode 2) Path to dump_details/ dir. Reads log_probs from train_data/{id}_{rank}.pt instead of .npy files.",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Rank index for dump_details train_data files (default: 0).",
    )
    parser.add_argument(
        "--rollout", type=int, nargs="*", default=None,
        help="Rollout IDs to include (default: all). Example: --rollout 0 1",
    )
    parser.add_argument("--json", action="store_true", help="Also print JSON output.")
    args = parser.parse_args()

    save_dir: Path = args.save_dir
    rollout_ids: list[int] | None = args.rollout

    if args.dump_details is not None:
        print(f"Loading logprobs from dump_details {args.dump_details} (rank={args.rank}) ...")
        log_probs, rollout_log_probs = _load_logprobs_from_dump_details(
            args.dump_details, args.rank, rollout_ids
        )
    else:
        print(f"Loading logprobs from {save_dir} ...")
        log_probs, rollout_log_probs = _load_logprobs(save_dir, rollout_ids)
    print(f"  log_probs shape:         {log_probs.shape}")
    print(f"  rollout_log_probs shape: {rollout_log_probs.shape}")

    assert log_probs.shape == rollout_log_probs.shape, (
        f"Shape mismatch: {log_probs.shape} vs {rollout_log_probs.shape}"
    )

    r = _pearson(log_probs, rollout_log_probs)
    abs_diff = np.abs(log_probs - rollout_log_probs)
    mse = float(np.mean((log_probs - rollout_log_probs) ** 2))

    print("\n=== Logprob match ===")
    print(f"  tokens:        {log_probs.size}")
    print(f"  Pearson r:     {r:.6f}")
    print(f"  MSE:           {mse:.6e}")
    print(f"  mean |diff|:   {abs_diff.mean():.6e}")
    print(f"  max  |diff|:   {abs_diff.max():.6e}")
    print(f"  p99  |diff|:   {np.percentile(abs_diff, 99):.6e}")

    megatron_hs = _load_megatron_hs(save_dir)
    if megatron_hs:
        print(f"\n=== Megatron hidden states ({len(megatron_hs)} layers) ===")
        _compute_hs_metrics(megatron_hs)
    else:
        print("\nNo Megatron hidden states found (run with --custom-megatron-before-log-prob-hook-path to enable).")

    csv_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_tokens": int(log_probs.size),
        "pearson_r": f"{r:.8f}",
        "mse": f"{mse:.6e}",
        "mean_abs_diff": f"{abs_diff.mean():.6e}",
        "max_abs_diff": f"{abs_diff.max():.6e}",
        "p99_abs_diff": f"{np.percentile(abs_diff, 99):.6e}",
    }
    csv_path = save_dir / "metrics" / "match.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
    print(f"\nResults appended to {csv_path}")

    if args.json:
        out = {k: v for k, v in csv_row.items()}
        print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
