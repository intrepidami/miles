# True-on-Policy Verification: SGLang vs Megatron Logprobs & Hidden States

## Goal

Capture per-token logprobs and per-layer hidden states from **both** SGLang (rollout/inference) and Megatron (training engine), then compare them to identify and eliminate train-inference mismatch.

## Current State: What Already Exists

| Capability | SGLang Side | Megatron Side | Compare Tool |
|---|---|---|---|
| Per-layer hidden state dump | ✅ `--dumper-enable` → `<dir>/engines/` | ✅ `--dumper-enable` → `<dir>/fwd_only/` | ✅ `sglang.srt.debug_utils.comparator --preset sglang_megatron` |
| Per-token logprob dump (JSON) | ❌ Not yet | ✅ `run_megatron --logprob-output` | ✅ `logprob_comparator.py` |
| Rollout sample logprobs | ✅ `Sample.rollout_log_probs` + `--dump-details` | N/A | ❌ No direct comparison tool |
| Training logprobs (RolloutBatch) | N/A | ✅ `--dump-details` → `train_data/{id}_{rank}.pt` | ❌ Different format from JSON |
| Standalone forward | SGLang generate endpoint | `run_megatron` CLI | `run-and-compare` (Megatron vs Megatron only) |

### Key gap

The SGLang dumper captures hidden states but **not** per-token logprobs in a format comparable with Megatron's `--logprob-output` JSON. The rollout logprobs live inside `Sample` objects in `.pt` dump files — a different shape than the `rank_N.json` format used by the Megatron comparison tool.

---

## Implementation Plan

### Phase 1: SGLang Logprob Dump → JSON

**Goal**: Make SGLang output per-token logprobs in the same JSON format as Megatron's standalone tool, so the existing `logprob_comparator.py` can diff them directly.

**What to build**:

1. **New file: `miles/rollout/generate_utils/sglang_logprob_dumper.py`**
   - Hook into the SGLang generate/prefill path to capture logits after the final layer
   - Compute `log_softmax` and extract per-token logprobs (same logic as `run_megatron/worker/output.py:_compute_logprob_entries`)
   - Save as `<dumper_dir>/engines/engine_{i}/logprobs/rank_{rank}.json`
   - Reuse the exact JSON schema: `{"rank": N, "tp_size": M, "cp_size": K, "logprob_entries": [[{global_position, token_id, logprob, is_valid}, ...]]}`

2. **Wire it into the dumper lifecycle**
   - SGLang's dumper already has `non_intrusive_mode` and `source_patcher_config` hooks for intercepting intermediate tensors
   - For logprobs we need the *final* logits, which may require a different hook point (post-lm-head)
   - Option A: register a forward hook on the lm_head in `get_sglang_env()` / `configure_sglang()`
   - Option B: add a source patcher config that wraps the generate endpoint to extract and save logprobs alongside the existing dumper

3. **Config flag**: `--dumper-logprobs` (bool, default true when `--dumper-enable` is set)
   - Controls whether logprobs are captured alongside hidden states during SGLang inference

**Files touched**:
- `miles/rollout/generate_utils/sglang_logprob_dumper.py` (new)
- `miles/utils/dumper_utils.py` — add `get_sglang_logprob_env()` / `configure_sglang_logprob()`
- `miles/utils/arguments.py` — add `--dumper-logprobs` flag

---

### Phase 2: Megatron Logprob Dump in Full Pipeline

**Goal**: In the full Miles training pipeline, when Megatron computes logprobs via `forward_only(get_log_probs_and_entropy, ...)`, also save them as JSON for comparison.

**What to build**:

1. **New file: `miles/backends/megatron_utils/logprob_dumper.py`**
   - Wraps `get_log_probs_and_entropy` to also dump logprobs as JSON
   - Reuses `run_megatron/worker/output.py:compute_and_save_output_info`
   - Saved to `<dumper_dir>/fwd_only/logprobs/rank_{rank}.json`
   - Only active when `--dumper-enable` + `--dumper-fwd-only enable=true`

2. **Wire into `forward_only()`**
   - In `miles/backends/megatron_utils/model.py:forward_only()`, after collecting logprobs via the `f` callback, call the new dumper if enabled
   - Need access to labels and position_ids (already in the batch from `get_batch`)

**Files touched**:
- `miles/backends/megatron_utils/logprob_dumper.py` (new)
- `miles/backends/megatron_utils/model.py` — add logprob dump after collector in `forward_only()`
- `miles/utils/dumper_utils.py` — add `configure_megatron_logprobs()` helper

---

### Phase 3: Unified Comparison CLI

**Goal**: A single command that runs SGLang standalone + Megatron standalone (via `run_megatron`), captures everything, then runs both activation and logprob comparisons.

**What to build**:

1. **New module: `miles/utils/debug_utils/verify_top/`** (or extend `run_megatron` CLI)

   ```
   miles/utils/debug_utils/verify_top/
   ├── __init__.py
   ├── __main__.py          # python -m miles.utils.debug_utils.verify_top
   ├── cli.py               # typer app with `verify` command
   ├── sglang_runner.py     # Launch SGLang, send generate request, collect dumps
   ├── megatron_runner.py   # Reuse run_megatron worker for Megatron forward
   └── comparator.py        # Orchestrate activation + logprob comparison
   ```

2. **`verify` command**:

   ```bash
   python -m miles.utils.debug_utils.verify_top verify \
       --model-type qwen3-4B \
       --hf-checkpoint /path/to/Qwen3-4B \
       --output-dir /tmp/top_verify \
       --prompt "What is 2+2?" \
       --tp 2 --cp 4 \
       --compare-logprobs \
       --logprob-threshold 1e-6
   ```

   Flow:
   ```
   Tokenize prompt
     ├── Step 1: SGLang standalone
     │   ├── Launch SGLang engine with dumper enabled
     │   ├── Send generate request (temperature=0, max_tokens=N)
     │   ├── Collect hidden states → output_dir/sglang/engines/
     │   └── Collect logprobs → output_dir/sglang/logprobs/
     │
     ├── Step 2: Megatron standalone (reuse run_megatron)
     │   ├── torchrun Megatron forward with same token IDs
     │   ├── DUMPER_ENABLE=1 → hidden states → output_dir/megatron/
     │   └── --logprob-output → logprobs → output_dir/megatron/logprobs/
     │
     └── Step 3: Compare
         ├── sglang.srt.debug_utils.comparator --preset sglang_megatron
         │     baseline=output_dir/sglang/engines/  target=output_dir/megatron/fwd_only/
         └── logprob_comparator.compare_logprobs
               baseline=output_dir/sglang/logprobs/  target=output_dir/megatron/logprobs/
   ```

3. **SGLang standalone runner** (`sglang_runner.py`):
   - Launch a temporary SGLang engine (like `RolloutServer` does but minimal)
   - Set dumper env vars (`DUMPER_ENABLE=1`, `DUMPER_DIR=...`, `DUMPER_SERVER_PORT=reuse`)
   - Send a single generate request with `return_logprob=True`, `temperature=0`
   - Collect the response logprobs + dumper output
   - Shut down the engine

4. **Megatron runner**: Reuse existing `run_megatron run` under the hood (already supports `--logprob-output` + dumper)

**Files touched**:
- `miles/utils/debug_utils/verify_top/` (new module)
- `miles/utils/debug_utils/run_megatron/cli/commands/run.py` — may need minor tweaks for programmatic reuse

---

### Phase 4 (Stretch): Full Pipeline Capture Mode

**Goal**: For production-scale validation, capture both sides from a single Miles training run.

This is a thin wrapper around Phase 1+2. Once SGLang and Megatron both dump logprobs alongside hidden states during normal pipeline execution, the full pipeline with `--dumper-enable --dumper-logprobs --dump-details` will produce:

```
<dumper_dir>/
├── engines/engine_0/
│   ├── standalone/          # SGLang hidden states (existing)
│   └── logprobs/            # SGLang logprobs (Phase 1)
│       └── rank_0.json
└── fwd_only/
    ├── standalone/           # Megatron hidden states (existing)
    └── logprobs/             # Megatron logprobs (Phase 2)
        └── rank_0.json

<dump_details>/
├── rollout_data/0.pt        # Original samples with rollout_log_probs
└── train_data/0_0.pt        # RolloutBatch with train log_probs
```

Then the comparison (Phase 3) can be run against these directories:

```bash
python -m miles.utils.debug_utils.verify_top compare \
    --baseline-dir <dumper_dir>/engines/engine_0 \
    --target-dir <dumper_dir>/fwd_only \
    --baseline-logprob-dir <dumper_dir>/engines/engine_0/logprobs \
    --target-logprob-dir <dumper_dir>/fwd_only/logprobs
```

---

### Phase 5 (Stretch): Rollout vs Train Logprob Comparison from `.pt` Dumps

**Goal**: Directly compare `Sample.rollout_log_probs` (from SGLang) with training `log_probs` (from Megatron `RolloutBatch`) without requiring the dumper JSON path.

This is useful for verifying that the existing `--recompute-logprobs-via-prefill` flag produces the same logprobs as Megatron's `forward_only(get_log_probs_and_entropy, ...)`.

**What to build**:

1. **New script: `miles/utils/debug_utils/compare_rollout_vs_train_logprobs.py`**
   - Load `rollout_data/{rollout_id}.pt` → extract `Sample.rollout_log_probs`
   - Load `train_data/{rollout_id}_{rank}.pt` → extract `RolloutBatch.log_probs`
   - Align by sample index and token position
   - Compute diffs and print report (reuse `logprob_comparator._ComputeResult` format)

**Files touched**:
- `miles/utils/debug_utils/compare_rollout_vs_train_logprobs.py` (new)

---

## Implementation Order (Recommended)

| Order | Phase | Effort | Rationale |
|---|---|---|---|
| 1 | Phase 1 + 2 (logprob dump on both sides) | Medium | Unblocks all comparison work — without this, there's nothing to diff |
| 2 | Phase 3 (unified CLI + SGLang runner) | Medium | Makes the workflow usable; validates Phase 1+2 end-to-end |
| 3 | Phase 4 (full pipeline capture) | Small | Thin wrapper, mostly documentation + minor wiring |
| 4 | Phase 5 (.pt dump comparison) | Small | Useful for CI and existing data, independent of dumper path |

## Design Decisions

### Why not add logprob capture to SGLang's dumper hook system directly?

SGLang's non-intrusive dumper hooks (`register_non_intrusive_dumper`) intercept intermediate activations *during* the model forward pass. The final logits/logprobs are computed after the model forward (in the sampling/logprob computation code). So we need a different hook point — either a source patch on the generate endpoint or a post-processing step after the generate call returns.

The simpler approach: after the generate request completes (with `return_logprob=True`), the logprobs are already in the response. Phase 1 just needs to extract them and save in the JSON format. No source patching needed for the basic case.

### Why use standalone tools instead of the full pipeline for Phase 3?

- **Determinism**: Standalone tools give a controlled environment — same input, same model weights, no Ray scheduling noise
- **Speed**: No placement group allocation, no weight sync overhead
- **Simplicity**: Each side is a single torchrun command

### JSON format alignment

The logprob JSON format from `run_megatron/worker/output.py`:
```json
{
  "rank": 0,
  "tp_size": 2,
  "cp_size": 4,
  "logprob_entries": [
    [
      {"global_position": 0, "token_id": 1234, "logprob": -1.23, "is_valid": true},
      {"global_position": 1, "token_id": 5678, "logprob": -0.45, "is_valid": true}
    ]
  ]
}
```

SGLang's logprobs from prefill scoring are indexed differently (by response token position, not global sequence position). Phase 1 must normalize to the same `global_position` scheme by computing `prompt_len + response_token_index`.

### Hidden state comparison path mapping

- SGLang dumper writes per-engine: `<dir>/engines/engine_{i}/standalone/`
- Megatron dumper writes per-phase: `<dir>/fwd_only/standalone/`
- The `sglang_megatron` preset in `sglang.srt.debug_utils.comparator` handles the internal format differences (layer naming, tensor shapes, dtype)
- We just need to pass the right paths

## Open Questions

1. **Token alignment**: SGLang generates autoregressively (one token at a time), while Megatron does a single forward pass over the full sequence. For hidden states, we can compare the prompt tokens (which both sides process in the same order). For logprobs, we compare response-token logprobs. Need to ensure the same prompt and same generation parameters (temperature=0, same max_tokens).

2. **Numerical tolerance**: What threshold is "good enough"? BF16 accumulation differences across different batch sizes / parallel configurations can cause ~1e-8 differences. The existing CI checks use 1e-8 for logprobs. Should start with 1e-6 and tighten based on findings.

3. **MoE models**: R3 (routing replay) adds complexity — the expert routing must be identical between SGLang and Megatron for a meaningful comparison. The `run_megatron` tool supports `--routing-replay-dump-path` / `--routing-replay-load-path` for this. Phase 3 should support a `--routing-replay` flag.
