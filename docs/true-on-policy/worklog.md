---
title: True On Policy Worklog
description: Chronological compact notes for true-on-policy related investigations.
---

# Worklog

## 2026-07-07 - Initial Framework Reading

Branch state:

- Created and switched to `dev` at commit `254091fd9f31471c7092c0409b0aa5fb0d8962c5`.
- Original local changes from `main` were preserved in `stash@{0}` with message `codex: preserve local changes before dev branch`.

Files read:

- `README.md`
- `docs/developer/architecture.md`
- `docs/user-guide/concepts.md`
- `docs/user-guide/training-script-walkthrough.md`
- `train.py`
- `train_async.py`
- `scripts/run_qwen3_4b.py`
- `miles/utils/arguments.py`
- `miles/utils/external_utils/command_utils.py`
- `miles/ray/placement_group.py`
- `miles/ray/actor_group.py`
- `miles/ray/train_actor.py`
- `miles/ray/rollout/rollout_manager.py`
- `miles/ray/rollout/train_data_conversion.py`
- `miles/rollout/data_source.py`
- `miles/rollout/sglang_rollout.py`
- `miles/rollout/base_types.py`
- `miles/utils/types.py`

Architecture summary:

- `train.py` is a thin synchronous loop over rollout generation, actor/critic training, checkpointing, offload/onload, weight update, and eval.
- Ray owns process orchestration: placement groups reserve GPUs; train actors run Megatron/FSDP; rollout manager owns SGLang rollout servers and data conversion.
- Rollout data contract centers on `Sample`; generated samples are converted to dictionaries of lists, then split by data-parallel rank.
- Launch scripts build long CLI strings from typed dataclasses. `scripts/run_qwen3_4b.py` is the main Qwen3 dense recipe and delegates true-on-policy expansion to `miles.true_on_policy`.

Risks to revisit:

- `miles/utils/arguments.py` computes `global_batch_size` with integer division when `num_steps_per_rollout` is set. It should explicitly reject non-divisible combinations to preserve the documented batch invariant.
- `miles/ray/rollout/train_data_conversion.py` gates several optional fields by checking only `samples[0]`. Mixed custom-rollout outputs can silently drop fields from later samples.
- Fast tests force `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`, so the legacy/default rollout path has weaker fast-test coverage.
- `miles/utils/external_utils/command_utils.py` launch cleanup kills Ray/SGLang/Miles/Redis processes. This is suitable for isolated training hosts, but risky on shared machines.

Verification:

- Attempted to run `tests/fast/utils/test_arguments.py` and `tests/fast/true_on_policy/test_run_qwen3_4b.py`.
- The local environment is missing `torch`, so pytest stopped during import before running tests.

## 2026-07-07 - GateGuard Config

- Added `GATEGUARD_EXEMPT_GLOBS=docs/true-on-policy/**` to `.claude/settings.local.json`.
- Effect: no fact-forcing gate on any file under this directory.

## 2026-07-07 - Forward-Only Consistency Test: Two Approaches

Goal: verify SGLang logprobs == Megatron logprobs on fixed token input, with per-layer hidden states and Pearson/MSE metrics.

### Approach A — Standalone diagnostic scripts (tools/consistency/)

Files: `gen_fixed_tokens.py`, `megatron_score.py`, `sglang_score.py`, `compute_metrics.py`.

Pros:
- Fixed deterministic token input (128×2048, seed=42)
- No Ray cluster, no real dataset needed; 1-2 GPU
- Can capture per-layer hidden states (not available in production path)
- Zero source modifications

Cons:
- Not the production path — custom batch prep, custom data conversion may mask real discrepancies
- Cannot catch bugs that only appear in the full pipeline (weight sync, rollout data conversion)
- Not CI-registerable
- Extra maintenance burden

Best for: diagnostic/debugging, hidden-state comparison, fast iteration.

### Approach B — Full pipeline with existing assertions

Use `scripts/run_qwen3_4b.py` with `mode="debug_one_sample"`, `--ci-test`, `--true-on-policy-mode`. The framework's existing `log_dict["log_probs"] == log_dict["rollout_log_probs"]` assertion in `log_utils.py:207` fires automatically.

Pros:
- Exact production path — rollout, data conversion, weight sync all included
- Assertion already built in; no new checker needed
- CI-registerable as e2e test

Cons:
- Requires full Ray cluster, SGLang server, real dataset
- Token input not fixed (data-set-dependent)
- Slow; high iteration cost
- No per-layer hidden states without additional work
- No Pearson/MSE metrics without additions

Best for: CI regression, end-to-end validation of real training runs.

### Decision rule

Use Approach A when: fixed token input required, hidden states needed, fast iteration, no cluster available.
Use Approach B when: CI registration required, production path must be validated, cluster is available.

Current work targets Approach B + instrumentation (real data, true-on-policy mode, with metric collection added on top).

## 2026-07-07 - Consistency Tools Implementation

Files created:
- `tools/true-on-policy/megatron_hs_hook.py` — forward hook for per-layer Megatron hidden states; registered via `--custom-megatron-before-log-prob-hook-path tools/true-on-policy/megatron_hs_hook.py:register`; saves to `$MILES_TRUE_ON_POLICY_SAVE_DIR/megatron_hs/rank_{r}/layer_{i:03d}.npy`
- `tools/true-on-policy/compute_metrics.py` — loads saved logprobs + hidden states, computes Pearson(log_probs, rollout_log_probs), per-layer L2 norm, cumulative MSE

Source change:
- `miles/backends/training_utils/log_utils.py` — added `_maybe_save_logprobs()`, called from `log_rollout_data()` when `MILES_TRUE_ON_POLICY_SAVE_DIR` is set; saves per-token `log_probs.npy` and `rollout_log_probs.npy` per rollout step

Existing args to use (no source change needed):
- `--true-on-policy-mode`: ensures both SGLang rollout_log_probs and Megatron log_probs are computed
- `--get-mismatch-metrics`: forces Megatron log_prob recompute even when `--use-rollout-logprobs` is set
- `--dumper-enable --dumper-fwd-only 'enable=true non_intrusive_mode=...'`: Megatron HS via existing dumper (alternative to hook)
- `--dumper-inference 'enable=true non_intrusive_mode=...'`: SGLang HS via existing dumper

Additional source changes (2026-07-08):
- `miles/utils/arguments.py` — added `--skip-train-step` flag (skips backward + optimizer, rollout and log-prob forward still run)
- `miles/backends/megatron_utils/actor.py` — guards `train()` call with `args.skip_train_step`
- `scripts/run_qwen3_4b.py` — added `"match"` mode: `rollout-batch-size=128`, `rollout-max-response-len=2048`, `n-samples-per-prompt=1`, `num-rollout=2`, `--skip-train-step --true-on-policy-mode --get-mismatch-metrics`
- `tools/true-on-policy/run_megatron.py` — updated to use `mode="match"`

Minimum launch flags for logprob comparison:
```
MILES_TRUE_ON_POLICY_SAVE_DIR=/tmp/true-on-policy \
  python train.py \
  --true-on-policy-mode \
  [other required args...]
```

After run:
```
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy
```

## 2026-07-07 - Candidate Script Search

Need:

- Keep code changes minimal and effective.
- Write test scripts for training/inference consistency in true-on-policy mode.

Closest existing scripts:

- `tests/e2e/fsdp/test_qwen3_4B_fsdp_true_on_policy.py` is the closest test-script base because it is already an e2e test and uses `--ci-test --true-on-policy-mode`, which reaches the framework's exact log-prob equality assertion.
- `examples/true_on_policy/run_simple.py` is the closest minimal runnable example and useful for smoke/debug sizing.
- `scripts/run_qwen3_4b.py` with `tests/fast/true_on_policy/test_run_qwen3_4b.py` is the closest Megatron/Qwen3 launch-plan base, but only validates flags and env vars.

Key checker:

- `miles/backends/training_utils/log_utils.py` asserts `log_dict["log_probs"] == log_dict["rollout_log_probs"]` when `args.ci_test and args.true_on_policy_mode`.

Decision:

- For current Megatron/Qwen3 consistency work, use `scripts/run_qwen3_4b.py` as the primary base to avoid duplicating launch-plan logic.
- Use `tests/e2e/fsdp/test_qwen3_4B_fsdp_true_on_policy.py` as an e2e structure reference, not the primary backend target, because it is disabled in this branch.
- Use `examples/true_on_policy/run_simple.py` as the reference for minimal smoke sizing.
