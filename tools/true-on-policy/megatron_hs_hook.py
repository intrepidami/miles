"""Hook for capturing per-layer Megatron hidden states during the forward-only log-prob pass.

Register via:
    --custom-megatron-before-log-prob-hook-path tools/true-on-policy/megatron_hs_hook.py:register

Requires MILES_TRUE_ON_POLICY_SAVE_DIR to be set. Saves one file per layer per rank:
    {MILES_TRUE_ON_POLICY_SAVE_DIR}/megatron_hs/rank_{r}/layer_{i:03d}.npy

Each file has shape [total_tokens, hidden_dim] (sequence-first layout transposed to token-first).
"""

from __future__ import annotations

import atexit
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

_store: dict[int, list[np.ndarray]] = {}  # layer_idx -> list of [seq, batch, hidden] arrays
_hooks: list[Any] = []


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _save() -> None:
    save_dir = os.environ.get("MILES_TRUE_ON_POLICY_SAVE_DIR")
    if not save_dir or not _store:
        return
    rank = _get_rank()
    hs_dir = Path(save_dir) / f"megatron_hs/rank_{rank}"
    hs_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, arrays in sorted(_store.items()):
        # Each array: [seq_len, batch, hidden] (Megatron sequence-first)
        # Stack microbatches along batch dim, then reshape to [total_tokens, hidden]
        combined = np.concatenate(arrays, axis=1)  # [seq, total_batch, hidden]
        seq, total_batch, hidden = combined.shape
        token_first = combined.transpose(1, 0, 2).reshape(total_batch * seq, hidden)
        np.save(hs_dir / f"layer_{layer_idx:03d}.npy", token_first)
    print(f"[megatron_hs_hook] Saved {len(_store)} layers → {hs_dir}", flush=True)


def register(args: Namespace, model: list[torch.nn.Module], store_prefix: str) -> None:
    """Called by the framework before the log-prob forward pass."""
    if not os.environ.get("MILES_TRUE_ON_POLICY_SAVE_DIR"):
        return

    model_chunk = model[0]
    layer_idx = 0

    for _name, module in model_chunk.named_modules():
        if hasattr(module, "self_attention") and hasattr(module, "mlp"):
            idx = layer_idx

            def _hook(m: torch.nn.Module, inp: Any, out: Any, _idx: int = idx) -> None:
                hs = out[0] if isinstance(out, tuple) else out
                _store.setdefault(_idx, []).append(hs.detach().float().cpu().numpy())

            _hooks.append(module.register_forward_hook(_hook))
            layer_idx += 1

    if layer_idx == 0:
        print("[megatron_hs_hook] WARNING: no transformer layers found via self_attention+mlp heuristic", flush=True)
    else:
        print(f"[megatron_hs_hook] Registered hooks on {layer_idx} layers", flush=True)

    atexit.register(_save)
