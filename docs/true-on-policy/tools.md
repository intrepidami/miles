---
title: True On Policy Consistency Tools
description: Files, launch instructions, and metrics for true-on-policy logprob and hidden-state consistency testing.
---
# Consistency Tools

## Files Changed

### Source modification

`miles/backends/training_utils/log_utils.py`

Added `_maybe_save_logprobs(rollout_data, rollout_id)`. Called from `log_rollout_data()` before any averaging. When `MILES_TRUE_ON_POLICY_SAVE_DIR` is set and the rank is TP=0, PP-last-stage, saves:

```
$MILES_TRUE_ON_POLICY_SAVE_DIR/
  rollout_0000/
    log_probs.npy          # float32 [total_tokens]  — Megatron recomputed logprobs
    rollout_log_probs.npy  # float32 [total_tokens]  — SGLang rollout logprobs
  rollout_0001/
    ...
```

### New files

`tools/true-on-policy/megatron_hs_hook.py`

Registers PyTorch `register_forward_hook` on every transformer layer (detected by `self_attention + mlp` attributes). Accumulates hidden states across microbatches and saves at process exit via `atexit`.

Output:

```
$MILES_TRUE_ON_POLICY_SAVE_DIR/
  megatron_hs/
    rank_0/
      layer_000.npy   # float32 [total_tokens, hidden_dim]
      layer_001.npy
      ...
```

`tools/true-on-policy/compute_metrics.py`

Loads saved logprobs and hidden states, computes and prints:

- Token count
- Pearson r (log_probs vs rollout_log_probs)
- MSE, mean |diff|, max |diff|, p99 |diff|
- Per-layer mean L2 norm of Megatron hidden states
- Cumulative MSE (mean of squared per-layer L2 norms)
- Optional JSON output (`--json`)

## Existing Args (no source change needed)

| Arg                                                           | Effect                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `--true-on-policy-mode`                                     | Megatron recomputes log_probs on rollout tokens; both logprob sets go into`rollout_data`       |
| `--get-mismatch-metrics`                                    | Forces Megatron log_prob recompute even when`--use-rollout-logprobs` is set                    |
| `--custom-megatron-before-log-prob-hook-path PATH:register` | Calls`register(args, model, store_prefix)` before log-prob forward; use for hidden state hooks |
| `--dumper-enable`                                           | Enables SGLang dumper for all phases                                                             |
| `--dumper-dir DIR`                                          | Base output directory for dumper                                                                 |
| `--dumper-fwd-only 'enable=true non_intrusive_mode=...'`    | Megatron forward-only HS via existing dumper (alternative to hook)                               |
| `--dumper-inference 'enable=true non_intrusive_mode=...'`   | SGLang inference HS via existing dumper                                                          |

## Launch

### Logprob comparison only (minimum)

```bash
export MILES_TRUE_ON_POLICY_SAVE_DIR=/tmp/true-on-policy

python train.py \
  --true-on-policy-mode \
  [model args] [data args] [parallelism args]
```

If `--use-rollout-logprobs` is also set, add `--get-mismatch-metrics` to force Megatron recompute.

### Logprob + Megatron hidden states

```bash
export MILES_TRUE_ON_POLICY_SAVE_DIR=/tmp/true-on-policy

python train.py \
  --true-on-policy-mode \
  --custom-megatron-before-log-prob-hook-path tools/true-on-policy/megatron_hs_hook.py:register \
  [model args] [data args] [parallelism args]
```

### Logprob + SGLang hidden states (via existing dumper)

```bash
export MILES_TRUE_ON_POLICY_SAVE_DIR=/tmp/true-on-policy

python train.py \
  --true-on-policy-mode \
  --dumper-enable \
  --dumper-dir /tmp/true-on-policy \
  --dumper-inference 'enable=true' \
  [model args] [data args] [parallelism args]
```

### All three (logprobs + both HS)

```bash
export MILES_TRUE_ON_POLICY_SAVE_DIR=/tmp/true-on-policy

python train.py \
  --true-on-policy-mode \
  --custom-megatron-before-log-prob-hook-path tools/true-on-policy/megatron_hs_hook.py:register \
  --dumper-enable \
  --dumper-dir /tmp/true-on-policy \
  --dumper-inference 'enable=true' \
  [model args] [data args] [parallelism args]
```

## Compute Metrics

```bash
# Basic report
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy

# Single rollout
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy --rollout 0

# JSON output
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy --json
```

## Notes

- `_maybe_save_logprobs` only runs on TP rank 0, PP last stage. With multi-rank setups, other ranks produce no files.
- `megatron_hs_hook.py` uses `atexit` for saving; reliable for single forward-only runs, but may miss data if the process is killed.
- Megatron hidden-state layout is sequence-first `[seq, batch, hidden]`; the hook transposes to `[total_tokens, hidden]` before saving.
- SGLang HS via `--dumper-inference` uses SGLang's own file format (not numpy); `compute_metrics.py` does not currently read it.
