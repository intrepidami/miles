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
import re
from collections import defaultdict
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


def _load_dumper_tensors(dump_dir: Path) -> dict[int, np.ndarray] | None:
    """Load dumper .pt files from a directory, group by layer_id, concat across steps.

    Returns dict: layer_id -> [total_tokens, hidden_dim] float32, or None if empty.
    Steps are sorted numerically before concat so token order is preserved.
    Tensors with ndim > 2 are flattened to [N, hidden_dim].
    """
    import torch

    if not dump_dir.exists():
        return None
    files = sorted(dump_dir.glob("*.pt"))
    if not files:
        return None

    grouped: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for f in files:
        layer_m = re.search(r"layer_id=(\d+)", f.stem)
        step_m = re.search(r"step=(\d+)", f.stem)
        if not layer_m:
            continue
        layer_id = int(layer_m.group(1))
        step = int(step_m.group(1)) if step_m else 0
        data = torch.load(f, map_location="cpu", weights_only=False)
        # file format: {"value": tensor, "meta": dict}
        t = data["value"] if isinstance(data, dict) and "value" in data else data
        # unwrap tuple/list (e.g. Megatron MLP returns (output, bias))
        while isinstance(t, (tuple, list)) and len(t) > 0:
            t = t[0]
        if not isinstance(t, torch.Tensor):
            continue
        arr = t.float().detach().numpy()
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        grouped[layer_id].append((step, arr))

    if not grouped:
        return None

    return {
        layer_id: np.concatenate([a for _, a in sorted(steps)], axis=0)
        for layer_id, steps in grouped.items()
    }


def _compare_hidden_states(save_dir: Path) -> None:
    """Compare Megatron fwd_only vs SGLang inference per-layer hidden states.

    Loads .pt files from tensor_cmp/fwd_only/ and tensor_cmp/engines/engine_0/.
    Tensors are grouped by layer_id and concatenated across steps in step order.
    Results are printed and appended to metrics/train_rollout_hidden_states.csv.

    Note on ordering: Megatron processes full sequences in batch; SGLang uses
    autoregressive KV-cache decoding (prefill step=0 then one token per step).
    After concat both have the same total token count, but token ORDER may differ
    if sequence lengths vary. MSE is meaningful only when shapes AND ordering match.
    """
    megatron_dir = save_dir / "tensor_cmp" / "fwd_only"
    sglang_dir = save_dir / "tensor_cmp" / "engines" / "engine_0"

    if not megatron_dir.exists() and not sglang_dir.exists():
        return

    print("\n=== Hidden state comparison (tensor_cmp) ===")

    megatron_hs = _load_dumper_tensors(megatron_dir)
    sglang_hs = _load_dumper_tensors(sglang_dir)

    if megatron_hs is None:
        print(f"  No Megatron tensors in {megatron_dir}")
        return
    if sglang_hs is None:
        print(f"  No SGLang tensors in {sglang_dir}")
        return

    common_layers = sorted(set(megatron_hs) & set(sglang_hs))
    if not common_layers:
        print("  No common layer_ids between Megatron and SGLang.")
        return

    print(f"  Megatron layers: {sorted(megatron_hs)}  SGLang layers: {sorted(sglang_hs)}")
    print(f"  {'layer':>5}  {'cosine_sim':>12}  {'mean_l2_m':>12}  {'mean_l2_s':>12}  {'mse':>12}  {'mean_L2_diff':>14}  megatron_shape / sglang_shape")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # always-computed columns first; shape-match-only (mse, mean_l2_diff) at the end
    csv_fields = ["timestamp", "layer_id", "megatron_shape", "sglang_shape", "cosine_sim", "mean_l2_megatron", "mean_l2_sglang", "mse", "mean_l2_diff"]
    csv_path = save_dir / "metrics" / "train_rollout_hidden_states.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    cosine_sims, l2_diffs, mses = [], [], []
    for layer_id in common_layers:
        m = megatron_hs[layer_id].astype(np.float64)
        s = sglang_hs[layer_id].astype(np.float64)
        mean_m = m.mean(axis=0)
        mean_s = s.mean(axis=0)
        denom = np.linalg.norm(mean_m) * np.linalg.norm(mean_s)
        cosine_sim = float(np.dot(mean_m, mean_s) / denom) if denom > 0 else float("nan")
        mean_l2_m = float(np.linalg.norm(m, axis=-1).mean())
        mean_l2_s = float(np.linalg.norm(s, axis=-1).mean())
        cosine_sims.append(cosine_sim)
        l2_diffs.append(abs(mean_l2_m - mean_l2_s))
        row = {
            "timestamp": timestamp, "layer_id": layer_id,
            "megatron_shape": str(m.shape), "sglang_shape": str(s.shape),
            "cosine_sim": f"{cosine_sim:.6f}",
            "mean_l2_megatron": f"{mean_l2_m:.6e}", "mean_l2_sglang": f"{mean_l2_s:.6e}",
            "mse": "", "mean_l2_diff": "",
        }
        if m.shape == s.shape:
            diff = m - s
            mse = float(np.mean(diff ** 2))
            mean_l2 = float(np.linalg.norm(diff, axis=-1).mean())
            mses.append(mse)
            row["mse"] = f"{mse:.6e}"
            row["mean_l2_diff"] = f"{mean_l2:.6e}"
            print(f"  {layer_id:>5}  {cosine_sim:>12.6f}  {mean_l2_m:>12.4e}  {mean_l2_s:>12.4e}  {mse:>12.4e}  {mean_l2:>14.4e}  {m.shape}")
        else:
            print(f"  {layer_id:>5}  {cosine_sim:>12.6f}  {mean_l2_m:>12.4e}  {mean_l2_s:>12.4e}  {'shape_mismatch':>12}  {'':>14}  {m.shape} / {s.shape}")
        rows.append(row)

    # cumulative summary
    mean_cosine = float(np.mean(cosine_sims)) if cosine_sims else float("nan")
    mean_l2_scale_diff = float(np.mean(l2_diffs)) if l2_diffs else float("nan")
    print(f"\n  Cumulative ({len(common_layers)} layers):")
    print(f"    mean cosine_sim:      {mean_cosine:.6f}")
    print(f"    mean |ΔL2_norm|:      {mean_l2_scale_diff:.4e}")
    if mses:
        cumulative_mse = float(np.mean(mses))
        print(f"    cumulative MSE:       {cumulative_mse:.6e}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {csv_path}")


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

    _compare_hidden_states(save_dir)

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
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)
    print(f"\nWritten {csv_path}")

    if args.json:
        out = {k: v for k, v in csv_row.items()}
        print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
