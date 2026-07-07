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
