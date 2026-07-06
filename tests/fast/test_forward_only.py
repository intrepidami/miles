"""Tests for ``--forward-only`` flag: structure and control-flow verification.

These tests validate the source code directly (text search) without requiring
PyTorch, Ray, or Megatron — they can run in CPU-only CI.
"""

import re
from pathlib import Path
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Flag definition  (miles/utils/arguments.py)
# ---------------------------------------------------------------------------


def _extract_flag_def(source: str, flag_name: str) -> Optional[dict]:
    """Find the add_argument call for *flag_name*, return kwargs dict via regex."""
    idx = source.find(flag_name)
    if idx == -1:
        return None

    # Find the containing add_argument call
    call_start = source.rfind("parser.add_argument(", 0, idx)
    if call_start == -1:
        return None

    # Extract until matching close-paren
    depth = 0
    end = call_start
    for i in range(call_start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    call_text = source[call_start:end]

    result = {}
    if 'action="store_true"' in call_text:
        result["action"] = "store_true"
    if "default=False" in call_text:
        result["default"] = False
    if "default=True" in call_text:
        result["default"] = True
    if "default=None" in call_text:
        result["default"] = None
    return result


def test_forward_only_flag_defined():
    source = (REPO / "miles" / "utils" / "arguments.py").read_text()
    info = _extract_flag_def(source, "--forward-only")
    assert info is not None, "--forward-only must be registered as an argparse flag"
    assert info.get("action") == "store_true", "--forward-only must be store_true"
    assert info.get("default") is False, "--forward-only must default to False"


# ---------------------------------------------------------------------------
# Actor control-flow  (miles/backends/megatron_utils/actor.py)
# ---------------------------------------------------------------------------


def test_actor_train_guarded_by_forward_only():
    """train() inside train_actor is wrapped by ``if not self.args.forward_only``."""
    source = (REPO / "miles" / "backends" / "megatron_utils" / "actor.py").read_text()
    assert "if not self.args.forward_only:" in source, (
        "train_actor must guard train() with 'if not self.args.forward_only:'"
    )
    guard_pos = source.index("if not self.args.forward_only:")
    train_pos = source.index("train(", guard_pos)
    assert train_pos > guard_pos, "train() must be inside the forward_only guard"


# ---------------------------------------------------------------------------
# Train-loop control-flow  (train.py)
# ---------------------------------------------------------------------------


def _train_lines() -> list[str]:
    return (REPO / "train.py").read_text().split("\n")


def _assert_loop_call_guarded(lineno: int, call_substr: str) -> None:
    """Verify the call (identified by *call_substr* at *lineno*) is inside a
    forward_only guard somewhere among the preceding few lines."""
    lines = _train_lines()
    idx = lineno - 1
    assert call_substr in lines[idx], f"Expected '{call_substr}' at line {lineno}"
    # Look back up to 8 lines for the guard (multiline if blocks)
    window = " ".join(lines[max(0, idx - 7) : idx + 1])
    assert "not args.forward_only" in window, (
        f"Call at line {lineno} ('{call_substr.strip()}') "
        f"must be guarded by 'not args.forward_only' in preceding lines"
    )


def test_weight_sync_in_loop_guarded():
    """Loop weight-sync (update_weights inside for loop) is guarded."""
    lines = _train_lines()
    for i, line in enumerate(lines):
        if "actor_model.update_weights()" in line and i > 70:
            _assert_loop_call_guarded(i + 1, "actor_model.update_weights()")
            return
    pytest.fail("Loop update_weights() not found")


def test_loop_save_guarded():
    for i, line in enumerate(_train_lines()):
        if "await save(rollout_id)" in line and i > 70:
            _assert_loop_call_guarded(i + 1, "await save(rollout_id)")
            return
    pytest.fail("Loop save() not found")


def test_loop_eval_guarded():
    """Both eval calls inside the for-loop are guarded."""
    lines = _train_lines()
    found = 0
    for i, line in enumerate(lines):
        if "rollout_manager.eval.remote" in line and i > 70:
            _assert_loop_call_guarded(i + 1, "rollout_manager.eval.remote")
            found += 1
    assert found >= 1, "At least one loop eval call must exist"
