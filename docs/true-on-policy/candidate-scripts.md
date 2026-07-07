---
title: True On Policy Candidate Scripts
description: Existing scripts closest to true-on-policy test-script work.
---

# Candidate Scripts

## Short Comparison

| Dimension | `scripts/run_qwen3_4b.py` | `examples/true_on_policy/run_simple.py` |
|---|---|---|
| Best for | Current Qwen3/Megatron true-on-policy work | Minimal FSDP true-on-policy smoke reference |
| Backend default | Megatron | FSDP |
| Public switch | `--true-on-policy` | Always true-on-policy |
| Testability | Best as a reusable launcher behind a thin e2e wrapper | Best as a standalone example |
| Consistency check path | `debug_one_sample` adds `--ci-test`; true-on-policy adds `--true-on-policy-mode` | Always adds `--ci-test` and `--true-on-policy-mode` |
| Launch complexity | Higher, but centralized and typed | Lower, but hardcoded |
| Model/data defaults | Qwen3-4B, dapo-math-17k, aime | Qwen3-0.6B, gsm8k |
| Existing fast coverage | `tests/fast/true_on_policy/test_run_qwen3_4b.py` checks generated args | No direct fast launch-plan test |
| Main risk | It is a launcher, not an e2e test file yet | It follows FSDP path, while this branch marks FSDP as experimental/not actively maintained |

Verdict:

- Use `scripts/run_qwen3_4b.py` if the target is the current framework path and Qwen3/Megatron parity.
- Use `examples/true_on_policy/run_simple.py` only to borrow the smallest smoke-run shape and the explicit flag set.

## Best Current Megatron Base

Use `scripts/run_qwen3_4b.py` as the closest base when the target is current Qwen3/Megatron true-on-policy testing.

Why:

- It owns the Qwen3 dense launch recipe currently under active development.
- It has a single public `true_on_policy` switch.
- It delegates derived flags and environment variables to `miles.true_on_policy`, avoiding duplicated launch-plan logic.
- `debug_one_sample` enables `--ci-test`, which reaches the framework's exact true-on-policy log-prob equality assertion.
- It already has fast tests for generated true-on-policy launch arguments in `tests/fast/true_on_policy/test_run_qwen3_4b.py`.

Limitation:

- It is a launcher, not a registered e2e test file. A CI test should wrap or reuse it rather than copy its full argument string.

## Existing E2E Test Base

Use `tests/e2e/fsdp/test_qwen3_4B_fsdp_true_on_policy.py` as the closest existing test-script base.

Why:

- It is already an e2e test file, not just an example.
- It launches true-on-policy mode with `--ci-test`.
- The framework already asserts exact equality for `log_probs` and `rollout_log_probs` when both `--ci-test` and `--true-on-policy-mode` are enabled.
- It includes the required deterministic SGLang/FSDP flags and environment variables.

Limitation:

- The file is currently registered as disabled because the FSDP backend is not actively maintained in this branch.

## Best Minimal Example Base

Use `examples/true_on_policy/run_simple.py` as the simplest runnable reference.

Why:

- It is intentionally described as the minimal true-on-policy example.
- It supports `Qwen3-0.6B` and `Qwen3-4B` via environment variables.
- It has `debug_one_sample` mode for short smoke runs.
- It contains the smallest visible set of true-on-policy flags.

Limitation:

- It is an example script, not a test registered with CI.

## Best Megatron/Qwen3 Launch-Plan Base

Use `scripts/run_qwen3_4b.py` plus `tests/fast/true_on_policy/test_run_qwen3_4b.py` when the test target is launch-plan correctness for Megatron.

Why:

- `scripts/run_qwen3_4b.py` owns the current Qwen3 true-on-policy launch-plan expansion.
- `tests/fast/true_on_policy/test_run_qwen3_4b.py` already checks the generated Megatron/SGLang flags and environment variables.

Limitation:

- The fast test only validates generated arguments; it does not run rollout/training log-prob equality.

## Current Recommendation

For a current Megatron/Qwen3 true-on-policy consistency test with minimal new code:

1. Reuse `scripts/run_qwen3_4b.py` as the source of launch arguments.
2. Use `mode="debug_one_sample"` so the run is short and `--ci-test` is enabled.
3. Prefer `Qwen3-0.6B` for the first smoke test if a one-GPU path is acceptable.
4. Add only a thin e2e test wrapper if CI registration is needed.
5. Keep `tests/e2e/fsdp/test_qwen3_4B_fsdp_true_on_policy.py` as a reference for e2e structure, not as the primary backend target.
