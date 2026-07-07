---
title: True On Policy Code Map
description: Compact map of relevant Miles files for true-on-policy work.
---

# Code Map

## Entry Points

- `train.py` - synchronous rollout/train loop. Creates placement groups, rollout manager, training actors, runs generate/train/save/update/eval.
- `train_async.py` - asynchronous variant. Starts the next rollout before training current data; does not support colocation.
- `scripts/run_qwen3_4b.py` - Python launch recipe for Qwen3 dense models, including true-on-policy launch-plan expansion.

## Configuration

- `miles/utils/arguments.py` - central CLI registration and validation. Important sections:
  - train backend selection and backend-specific parser dispatch
  - rollout/data/eval arguments
  - true-on-policy and chat-template validation
  - batch-size invariant checks
- `miles/true_on_policy/` - true-on-policy contracts, model profiles, and launch-plan defaults.

## Ray Orchestration

- `miles/ray/placement_group.py` - GPU placement, actor/critic group construction, rollout manager construction.
- `miles/ray/actor_group.py` - RayTrainGroup wrapper around train actor ranks.
- `miles/ray/train_actor.py` - base train actor process setup, distributed init, and rollout manager connection.
- `miles/ray/rollout/rollout_manager.py` - rollout lifecycle, conversion to train data, eval, offload/onload, health monitor integration.

## Rollout And Data Contract

- `miles/rollout/sglang_rollout.py` - legacy/default SGLang rollout path.
- `miles/rollout/inference_rollout/` - experimental rollout refactor path; fast tests enable this by default.
- `miles/rollout/base_types.py` - rollout function input/output dataclasses.
- `miles/utils/types.py` - `Sample` contract and validation helpers.
- `miles/ray/rollout/train_data_conversion.py` - converts generated `Sample` objects into per-DP training batches.
- `miles/rollout/data_source.py` - prompt dataset and buffer management.

## Training Objective

- `miles/backends/training_utils/loss.py` - dispatch and scaling for policy/value/SFT/custom loss.
- `miles/backends/training_utils/loss_hub/` - advantage estimators, corrections, logit processing, and concrete loss functions.

## Tests

- `tests/fast/true_on_policy/` - launch-plan and true-on-policy config tests.
- `tests/fast/utils/test_arguments.py` - parser extension and selected validation tests.
- `tests/e2e/precision/` - precision and alignment oriented checks.
- `tests/e2e/megatron/` - full Megatron/SGLang integration coverage.

