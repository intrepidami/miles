"""Compute true-on-policy consistency metrics from saved logprobs and hidden states.

Reads output from a pipeline run with:
    MILES_TRUE_ON_POLICY_SAVE_DIR=<save_dir>
    --true-on-policy-mode (or --get-mismatch-metrics)
    --custom-megatron-before-log-prob-hook-path tools/true-on-policy/megatron_hs_hook.py:register

Usage:
    python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy
    python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy --rollout 0

Metrics computed:
    - Pearson correlation of log_probs vs rollout_log_probs (per token)
    - Per-layer MSE of Megatron hidden states across rollouts (if available)
    - Cumulative MSE across layers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--rollout", type=int, nargs="*", default=None, help="Rollout IDs to load (default: all)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    save_dir: Path = args.save_dir
    rollout_ids: list[int] | None = args.rollout

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

    print("\n=== Logprob consistency ===")
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

    if args.json:
        out = {
            "num_tokens": int(log_probs.size),
            "pearson_r": r,
            "mse": mse,
            "mean_abs_diff": float(abs_diff.mean()),
            "max_abs_diff": float(abs_diff.max()),
            "p99_abs_diff": float(np.percentile(abs_diff, 99)),
        }
        print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
