"""Compute true-on-policy match metrics from saved logprobs and hidden states.

Mode is auto-detected from --save-dir:

  Mode 2 — dump_details (preferred, auto-detected):
    Used when <save_dir>/dump_details/train_data/ exists.
    Reads {rollout_id}_{rank}.pt, extracts log_probs and rollout_log_probs from RolloutBatch.

  Mode 1 — .npy files (fallback):
    Used when dump_details is absent.
    Reads rollout_*/log_probs.npy and rollout_*/rollout_log_probs.npy.

CSV output always written to <save_dir>/metrics/match.csv.

Usage:
    python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022
    python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022 --rollout 0
    python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022 --rank 0
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


def _verify_sample_alignment(rb, lp_val, rlp_val, fname: str) -> None:
    """Verify per-sample train/rollout logprob alignment inside one RolloutBatch.

    A global size match after concat can hide per-sample misalignment (e.g. one
    sample longer, the next shorter). Checks, per sample i:
      - log_probs[i].shape == rollout_log_probs[i].shape
      - numel == response_lengths[i] (skipped when totals differ, i.e. CP-sharded)
    """
    if not (isinstance(lp_val, (list, tuple)) and isinstance(rlp_val, (list, tuple))):
        return
    assert len(lp_val) == len(rlp_val), (
        f"{fname}: sample count mismatch: {len(lp_val)} log_probs vs {len(rlp_val)} rollout_log_probs"
    )
    for i, (lp, rlp) in enumerate(zip(lp_val, rlp_val)):
        assert lp.shape == rlp.shape, (
            f"{fname}: sample {i}: log_probs {tuple(lp.shape)} vs rollout_log_probs {tuple(rlp.shape)}"
        )

    resp = rb.get("response_lengths") if hasattr(rb, "get") else getattr(rb, "response_lengths", None)
    if resp is None:
        print(f"  {fname}: no response_lengths; skipped response-length check")
        return
    total_tokens = sum(lp.numel() for lp in lp_val)
    total_resp = int(sum(int(r) for r in resp))
    if total_tokens != total_resp:
        # CP > 1 stores per-rank shards; per-sample numel != response_length by design
        print(
            f"  {fname}: total tokens {total_tokens} != sum(response_lengths) {total_resp} "
            f"(CP-sharded?); skipped per-sample response-length check"
        )
        return
    assert len(resp) == len(lp_val), (
        f"{fname}: {len(resp)} response_lengths vs {len(lp_val)} samples"
    )
    for i, (lp, r) in enumerate(zip(lp_val, resp)):
        assert lp.numel() == int(r), (
            f"{fname}: sample {i}: log_probs has {lp.numel()} tokens, response_length is {int(r)}"
        )
    print(f"  {fname}: alignment OK ({len(lp_val)} samples, {total_tokens} tokens match response_lengths)")


def _load_logprobs_from_dump_details(
    dump_details_dir: Path, rank: int, rollout_ids: list[int] | None
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]] | None]:
    import torch

    train_data_dir = dump_details_dir / "train_data"
    if rollout_ids is not None:
        files = sorted(train_data_dir / f"{r}_{rank}.pt" for r in rollout_ids)
    else:
        files = sorted(train_data_dir.glob(f"*_{rank}.pt"))

    if not files:
        raise FileNotFoundError(f"No train_data files for rank={rank} in {train_data_dir}")

    lp_parts, rlp_parts = [], []
    length_pairs: list[tuple[int, int]] | None = []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        rb = data["rollout_data"]
        if length_pairs is not None:
            totals = rb.get("total_lengths") if hasattr(rb, "get") else getattr(rb, "total_lengths", None)
            resps = rb.get("response_lengths") if hasattr(rb, "get") else getattr(rb, "response_lengths", None)
            if totals is None or resps is None:
                length_pairs = None
            else:
                length_pairs.extend((int(t) - int(r), int(t)) for t, r in zip(totals, resps))
        # Collect both keys before appending so a file missing one key
        # cannot leave lp_parts and rlp_parts misaligned.
        vals = {}
        for key in ("log_probs", "rollout_log_probs"):
            val = rb.get(key) if hasattr(rb, "get") else getattr(rb, key, None)
            if val is None:
                print(f"  skip {f.name}: missing {key}")
                break
            vals[key] = val
        else:
            _verify_sample_alignment(rb, vals["log_probs"], vals["rollout_log_probs"], f.name)
            for key, parts in [("log_probs", lp_parts), ("rollout_log_probs", rlp_parts)]:
                val = vals[key]
                if isinstance(val, (list, tuple)) and val:
                    parts.append(torch.cat(val).float().numpy())
                else:
                    parts.append(val.float().numpy())

    if not lp_parts:
        raise FileNotFoundError(f"No valid train_data entries in {train_data_dir}")
    return np.concatenate(lp_parts), np.concatenate(rlp_parts), length_pairs


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
        lp = np.load(lp_path)
        rlp = np.load(rlp_path)
        # check per rollout, not only after global concat: equal totals can
        # hide two rollouts whose mismatches cancel out
        assert lp.shape == rlp.shape, (
            f"{rd.name}: log_probs {lp.shape} vs rollout_log_probs {rlp.shape}"
        )
        print(f"  {rd.name}: {lp.size} tokens, shapes match")
        lp_parts.append(lp)
        rlp_parts.append(rlp)

    if not lp_parts:
        raise FileNotFoundError(f"No valid rollout dirs found in {save_dir}")

    return np.concatenate(lp_parts), np.concatenate(rlp_parts)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
    return float(np.dot(a_c, b_c) / denom) if denom > 0 else float("nan")


def _load_dumper_files(dump_dir: Path) -> dict[int, list[tuple[int, np.ndarray]]] | None:
    """Load dumper .pt files, group by layer_id, keep per-file arrays with step order.

    Returns dict: layer_id -> [(step, arr), ...] sorted by step, original ndim
    preserved, or None if the directory is empty.
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
        grouped[layer_id].append((step, t.float().detach().numpy()))

    if not grouped:
        return None
    # sort by step only: tuple comparison would fall through to ndarray
    # comparison on equal steps and raise "truth value is ambiguous"
    return {lid: sorted(steps, key=lambda x: x[0]) for lid, steps in grouped.items()}


def _load_dumper_tensors(dump_dir: Path) -> dict[int, np.ndarray] | None:
    """Flattened view: layer_id -> [total_tokens, hidden_dim], concat across steps."""
    grouped = _load_dumper_files(dump_dir)
    if grouped is None:
        return None
    out = {}
    for layer_id, steps in grouped.items():
        arrs = [a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a for _, a in steps]
        out[layer_id] = np.concatenate(arrs, axis=0)
    return out


def _megatron_sample_seqs(
    steps: list[tuple[int, np.ndarray]], length_pairs: list[tuple[int, int]]
) -> list[np.ndarray] | None:
    """One dumper file per sample (match mode runs with --micro-batch-size 1).

    Validates file k has exactly total_lengths[k] tokens; returns per-sample
    [total_len, hidden] arrays in rollout-data order, or None with a reason.
    """
    if len(steps) != len(length_pairs):
        print(f"    megatron: {len(steps)} files != {len(length_pairs)} samples (need --micro-batch-size 1 run)")
        return None
    seqs = []
    for k, (_, arr) in enumerate(steps):
        if arr.ndim == 3:
            if arr.shape[1] != 1:
                print(f"    megatron: file {k} batch dim {arr.shape[1]} != 1 (dynamic batching?)")
                return None
            arr = arr[:, 0, :]
        total = length_pairs[k][1]
        if arr.shape[0] != total:
            print(f"    megatron: file {k} has {arr.shape[0]} tokens, expected total_length {total}")
            return None
        seqs.append(arr)
    return seqs


def _sglang_decode_stack(
    steps: list[tuple[int, np.ndarray]], num_samples: int, response_len: int
) -> np.ndarray | None:
    """Stack decode-step files into [T, N, hidden].

    Decode files have shape [N, hidden] (one token per running request).
    Prefill chunks have other shapes and are dropped. The dumper step counter
    increments once per forward pass, so decode files form one contiguous step
    run of length response_len-1 or response_len; prefill chunks that happen to
    have N rows land in other (shorter) runs and are discarded by taking the
    longest contiguous run.
    """
    cand = [(s, a) for s, a in steps if a.ndim == 2 and a.shape[0] == num_samples]
    if not cand:
        print("    sglang: no decode-shaped files")
        return None
    runs: list[list[np.ndarray]] = [[cand[0][1]]]
    for (prev_s, _), (s, a) in zip(cand, cand[1:]):
        if s == prev_s + 1:
            runs[-1].append(a)
        else:
            runs.append([a])
    run = max(runs, key=len)
    if not (response_len - 1 <= len(run) <= response_len):
        print(f"    sglang: longest decode run has {len(run)} steps, expected {response_len - 1} or {response_len}")
        return None
    return np.stack(run, axis=0)


def _match_decode_columns(
    meg_seqs: list[np.ndarray],
    sg_stack: np.ndarray,
    length_pairs: list[tuple[int, int]],
) -> list[int] | None:
    """Map SGLang decode column c -> sample index, by cosine of hidden states.

    Decode step t of a request processes its response token t, which sits at
    position prompt_len + t in the Megatron full-sequence forward. Matches on
    t=0, then verifies the assignment at t = T//2 and T-1.
    """
    n = len(meg_seqs)
    t_total = sg_stack.shape[0]

    def cos_matrix(t: int) -> np.ndarray:
        u = np.stack([meg_seqs[k][length_pairs[k][0] + t] for k in range(n)]).astype(np.float64)
        v = sg_stack[t].astype(np.float64)
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-30
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-30
        return v @ u.T  # [columns, samples]

    assign = np.argmax(cos_matrix(0), axis=1)
    if len(set(assign.tolist())) != n:
        print("    match: column->sample assignment not one-to-one")
        return None
    for t in (t_total // 2, t_total - 1):
        m = cos_matrix(t)
        matched = m[np.arange(n), assign]
        if matched.mean() < 0.9:
            print(f"    match: verification at t={t} failed (mean cosine {matched.mean():.4f})")
            return None
    return assign.tolist()


def _compare_hidden_states_aligned(save_dir: Path, length_pairs: list[tuple[int, int]]) -> bool:
    """Token-aligned per-layer comparison. Returns False if preconditions fail.

    Requires a match-mode run with --micro-batch-size 1 (one Megatron dumper
    file per sample, rollout order) and uniform response lengths (SGLang decode
    batch stays full-size). Column-to-sample mapping is recovered by cosine
    matching, so SGLang's internal batch order does not need to equal rollout order.
    """
    meg_files = _load_dumper_files(save_dir / "tensor_cmp" / "fwd_only")
    sg_files = _load_dumper_files(save_dir / "tensor_cmp" / "engines" / "engine_0")
    if meg_files is None or sg_files is None:
        return False
    common = sorted(set(meg_files) & set(sg_files))
    if not common:
        return False

    responses = {total - prompt for prompt, total in length_pairs}
    if len(responses) != 1:
        print(f"    aligned: response lengths vary ({sorted(responses)[:5]}...), decode batch not constant")
        return False
    response_len = responses.pop()
    n = len(length_pairs)

    # Match on the first common layer, then reuse the assignment everywhere.
    ref = common[0]
    ref_seqs = _megatron_sample_seqs(meg_files[ref], length_pairs)
    ref_stack = _sglang_decode_stack(sg_files[ref], n, response_len)
    if ref_seqs is None or ref_stack is None:
        return False
    assign = _match_decode_columns(ref_seqs, ref_stack, length_pairs)
    if assign is None:
        return False
    t_total = ref_stack.shape[0]
    print(f"\n=== Hidden state comparison (token-aligned, {n} samples x {t_total} decode steps) ===")
    print(f"  {'layer':>5}  {'mse':>12}  {'mean_L2_diff':>14}  {'mean_token_cos':>15}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for layer_id in common:
        seqs = ref_seqs if layer_id == ref else _megatron_sample_seqs(meg_files[layer_id], length_pairs)
        stack = ref_stack if layer_id == ref else _sglang_decode_stack(sg_files[layer_id], n, response_len)
        if seqs is None or stack is None:
            print(f"  {layer_id:>5}  skipped (load/validation failed)")
            continue
        sq_sum = l2_sum = cos_sum = 0.0
        count = 0
        for c in range(n):
            k = assign[c]
            prompt = length_pairs[k][0]
            m = seqs[k][prompt : prompt + stack.shape[0]].astype(np.float64)
            s = stack[:, c].astype(np.float64)
            diff = m - s
            sq_sum += float((diff**2).sum())
            l2_sum += float(np.linalg.norm(diff, axis=-1).sum())
            mn = np.linalg.norm(m, axis=-1) * np.linalg.norm(s, axis=-1)
            cos_sum += float(((m * s).sum(axis=-1) / (mn + 1e-30)).sum())
            count += m.shape[0]
        mse = sq_sum / (count * seqs[0].shape[-1])
        mean_l2 = l2_sum / count
        mean_cos = cos_sum / count
        print(f"  {layer_id:>5}  {mse:>12.4e}  {mean_l2:>14.4e}  {mean_cos:>15.6f}")
        rows.append({
            "timestamp": timestamp, "layer_id": layer_id, "num_tokens": count,
            "mse": f"{mse:.6e}", "mean_l2_diff": f"{mean_l2:.6e}", "mean_token_cosine": f"{mean_cos:.6f}",
        })

    if not rows:
        return False
    csv_path = save_dir / "metrics" / "train_rollout_hidden_states_aligned.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {csv_path}")
    return True


def _compare_hidden_states(save_dir: Path, length_pairs: list[tuple[int, int]] | None = None) -> None:
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

    if length_pairs is not None:
        if _compare_hidden_states_aligned(save_dir, length_pairs):
            return
        print("  token-aligned comparison unavailable, falling back to unaligned summary")

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


def _plot_logprob_scatter(
    log_probs: np.ndarray,
    rollout_log_probs: np.ndarray,
    pearson_r: float,
    save_dir: Path,
    svg: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    probs = np.exp(log_probs.astype(np.float64))
    rollout_probs = np.exp(rollout_log_probs.astype(np.float64))

    # --- leftmost: probs scatter ---
    ax0 = axes[0]
    ax0.scatter(rollout_probs, probs, s=1, alpha=0.4, linewidths=0)
    lo0 = min(probs.min(), rollout_probs.min())
    hi0 = max(probs.max(), rollout_probs.max())
    ax0.plot([lo0, hi0], [lo0, hi0], "r--", linewidth=1, label="y = x")
    ax0.set_xlabel("rollout_probs (SGLang)")
    ax0.set_ylabel("probs (Megatron)")
    ax0.set_title("Prob scatter")
    ax0.legend(fontsize=8)

    # --- middle: logprob scatter ---
    ax = axes[1]
    ax.scatter(rollout_log_probs, log_probs, s=1, alpha=0.4, linewidths=0)
    lo = min(log_probs.min(), rollout_log_probs.min())
    hi = max(log_probs.max(), rollout_log_probs.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("rollout_log_probs (SGLang)")
    ax.set_ylabel("log_probs (Megatron)")
    ax.set_title(f"Logprob scatter  Pearson r = {pearson_r:.6f}")
    ax.legend(fontsize=8)

    # --- right: histogram of |diff| (all tokens) ---
    ax2 = axes[2]
    abs_diff = np.abs(log_probs - rollout_log_probs)
    ax2.hist(abs_diff, bins=100, log=True, color="steelblue", edgecolor="none")
    ax2.set_xlabel("|log_probs − rollout_log_probs|")
    ax2.set_ylabel("count (log scale)")
    ax2.set_title(f"Abs diff  mean={abs_diff.mean():.2e}  p99={np.percentile(abs_diff, 99):.2e}")

    fig.tight_layout()
    metrics_dir = save_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if svg:
        out = metrics_dir / "logprob_scatter.svg"
        fig.savefig(out, format="svg", bbox_inches="tight")
    else:
        out = metrics_dir / "logprob_scatter.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out}")


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
        "--rank", type=int, default=0,
        help="Rank index for dump_details train_data files (default: 0).",
    )
    parser.add_argument(
        "--rollout", type=int, nargs="*", default=None,
        help="Rollout IDs to include (default: all). Example: --rollout 0 1",
    )
    parser.add_argument("--json", action="store_true", help="Also print JSON output.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metric", action="store_true", help="Compute metrics and write CSV only (default).")
    mode.add_argument("--plot", action="store_true", help="Generate logprob_scatter.png only.")
    mode.add_argument("--all", action="store_true", help="Compute metrics and generate plot.")
    parser.add_argument("--svg", action="store_true", help="Save plot as SVG instead of PNG.")
    args = parser.parse_args()

    do_metric = args.metric or args.all or (not args.plot)
    do_plot = args.plot or args.all

    save_dir: Path = args.save_dir
    rollout_ids: list[int] | None = args.rollout

    dump_details_dir = save_dir / "dump_details"
    length_pairs: list[tuple[int, int]] | None = None
    if (dump_details_dir / "train_data").exists():
        print(f"Loading logprobs from dump_details (rank={args.rank}) ...")
        log_probs, rollout_log_probs, length_pairs = _load_logprobs_from_dump_details(
            dump_details_dir, args.rank, rollout_ids
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

    if do_metric:
        print("\n=== Logprob match ===")
        print(f"  tokens:        {log_probs.size}")
        print(f"  Pearson r:     {r:.6f}")
        print(f"  MSE:           {mse:.6e}")
        print(f"  mean |diff|:   {abs_diff.mean():.6e}")
        print(f"  max  |diff|:   {abs_diff.max():.6e}")
        print(f"  p99  |diff|:   {np.percentile(abs_diff, 99):.6e}")

        _compare_hidden_states(save_dir, length_pairs)

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
            print("\n" + json.dumps(csv_row, indent=2))

    if do_plot:
        _plot_logprob_scatter(log_probs, rollout_log_probs, r, save_dir, svg=args.svg)


if __name__ == "__main__":
    main()
