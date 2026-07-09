# true-on-policy tools

Test that SGLang rollout logprobs match Megatron logprobs under `--true-on-policy-mode`.

## Files

| File                    | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `run_match.py`        | Runner: wraps`scripts/run_qwen3_4b.py` in `match` mode (megatron or fsdp backend) |
| `megatron_hs_hook.py` | Forward hook: captures per-layer Megatron hidden states    |
| `compute_metrics.py`  | Offline analysis: Pearson r, MSE, per-layer L2 norms       |

## Setup

```bash
pip install -e . --no-build-isolation --no-deps
```

Also ensure `MEGATRON_PATH` is on `PYTHONPATH`:

```bash
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH
```

## Config

Edit the `CONFIG` block at the top of `run_match.py` (paths only; runtime behavior is CLI flags):

```python
MODEL_NAME        = "Qwen3-0.6B"      # 1 GPU (TP=1); "Qwen3-4B" needs more GPUs
MODEL_DIR         = "/root/models"
DATA_DIR          = "/root/datasets"
OUTPUT_DIR        = "/root/output"
MEGATRON_PATH     = "/root/Megatron-LM"
BASE_DIR          = "/root/true-on-policy"   # each run saves to BASE_DIR/{YYYYMMDD_HHMMSS}/
```

## Run

First run — downloads Qwen3-0.6B + dapo-math-17k, converts HF checkpoint to Megatron format:

```bash
python tools/true-on-policy/run_match.py
```

Subsequent runs — skip download/conversion:

```bash
python tools/true-on-policy/run_match.py --skip-prepare
```

Megatron single GPU:

```bash
python tools/true-on-policy/run_match.py \
    --cuda-visible-devices 5 \
    --num-gpus-per-node 1 \
    --num-nodes 1 \
    --skip-prepare \
    --train-backend megatron \
    --true-on-policy
```

FSDP with 2 GPUs (DP=2):

```bash
python tools/true-on-policy/run_match.py \
    --cuda-visible-devices 4,5 \
    --num-gpus-per-node 2 \
    --skip-prepare \
    --train-backend fsdp \
    --true-on-policy
```

Optional capture flags (both default OFF — nothing extra is written to disk):

- `--capture-hidden-states` — Megatron per-layer hidden states via `megatron_hs_hook.py` (ignored with a warning on fsdp)
- `--dumper-enable` — SGLang/Megatron tensor dumper into `tensor_cmp/`

All output (stdout + stderr) is automatically tee'd to `log/log_YYYYMMDD_HHMMSS.txt` inside the run directory.

What each run does:

- Samples 128 prompts from dapo-math-17k, generates up to 2048 tokens, runs 1 rollout step
- Skips backward pass and optimizer (`--skip-train-step`)
- SGLang generates tokens and records `rollout_log_probs`
- The training backend recomputes `log_probs` on the same tokens
- Both arrays saved under `BASE_DIR/{YYYYMMDD_HHMMSS}/` as `.npy` files

## Analyze results

```bash
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/<YYYYMMDD_HHMMSS>
```

Add `--hidden-states` to also compare per-layer hidden states from `tensor_cmp/` (default: off; needs a run with `--dumper-enable`).

Expected output when logprobs match:

```
Loading logprobs from /root/true-on-policy/<YYYYMMDD_HHMMSS> ...
  log_probs shape:         (262144,)
  rollout_log_probs shape: (262144,)

=== Logprob match ===
  tokens:        262144
  Pearson r:     1.000000
  MSE:           0.000000e+00
  mean |diff|:   0.000000e+00
  max  |diff|:   0.000000e+00
  p99  |diff|:   0.000000e+00

=== Megatron hidden states (28 layers) ===
  layer   0: mean_L2=12.3456  shape=(262144, 1024)
  layer   1: mean_L2=14.7891  shape=(262144, 1024)
  ...
Cumulative MSE (mean of squared L2 norms): 198.3421
```

Pearson r = 1.0 and MSE = 0 means perfect logprob match. Any meaningful deviation indicates a true-on-policy bug.

Additional flags:

```bash
# JSON output
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy --json

# Single rollout step only
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy --rollout 0
```

## Saved file layout

```
/root/true-on-policy/
  rollout_0000/
    log_probs.npy           # [total_tokens] float32 — Megatron recomputed logprobs
    rollout_log_probs.npy   # [total_tokens] float32 — SGLang original logprobs
  rollout_0001/
    ...
  megatron_hs/
    rank_0/
      layer_000.npy         # [total_tokens, hidden_dim] float32
      layer_001.npy
      ...
```

## Source changes

| File                                           | Change                                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `miles/utils/arguments.py`                   | Added`--skip-train-step` flag                                                          |
| `miles/backends/megatron_utils/actor.py`     | Guards`train()` call with `args.skip_train_step`                                     |
| `miles/backends/training_utils/log_utils.py` | `_maybe_save_logprobs()` writes `.npy` when `MILES_TRUE_ON_POLICY_SAVE_DIR` is set |
| `scripts/run_qwen3_4b.py`                    | Added`match` mode                                                                      |

## Env vars

| Variable                          | Set by                            | Effect                                |
| --------------------------------- | --------------------------------- | ------------------------------------- |
| `MILES_TRUE_ON_POLICY_SAVE_DIR` | `run_match.py` automatically | Enables logprob + hidden-state saving |

## Troubleshooting

**`rollout_log_probs.npy` missing** — `--true-on-policy-mode` not active. Pass `--true-on-policy` to `run_match.py`.

**`log_probs.npy` missing** — worker not on TP rank=0 / PP last stage. Normal for multi-GPU; only one rank writes.

**No hidden states** — pass `--capture-hidden-states` (megatron backend only).

**Checkpoint conversion fails** — verify `MEGATRON_PATH` points to a working Megatron-LM install and is on `PYTHONPATH`.

**SGLang OOM** — reduce `--sglang-mem-fraction-static` (default 0.7 for Megatron backend, 0.75 for fsdp) via `extra_args` in `run_match.py`.
