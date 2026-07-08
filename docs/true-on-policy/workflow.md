---
title: AI 工作指南
description: 新会话 AI 从零掌握本项目 true-on-policy 工作所需的所有信息。
---

# AI 工作指南

## 项目背景

Miles 是一个基于 Ray 的 RL 训练框架。训练引擎用 Megatron，推理引擎用 SGLang。

**True-on-policy** 的核心要求：SGLang 生成 token 时记录的 `rollout_log_probs`，必须与 Megatron 对同一批 token 重算的 `log_probs` 数值完全一致。如果不一致，训练梯度用的是错误的 logprobs。

当前工作：构建并运行一个前向一致性测试，验证 `--true-on-policy-mode` 下两者是否匹配。

## 会话开始：从零上手

读以下三个文件，顺序不要乱：

1. `docs/true-on-policy/README.md` — 目标、当前状态、运行命令（本文件同目录）
2. `docs/true-on-policy/implementation.md` — 改了什么源码，工具怎么用
3. `docs/true-on-policy/code-map.md` — 关键代码在哪

如果需要理解框架架构，读 `docs/developer/architecture.md`。

不要在没读完上述文件的情况下开始修改代码。

## 工作规则

### 改动原则

- 最小改动。能加新文件就不改现有源文件。
- 改源文件前先确认：是否有现成机制可以用（hook、env var、flag）。
- 不做与当前任务无关的重构或清理。

### 提交规则

- 每个 commit 只做一件事（一个逻辑改动）。
- 用 `git add <specific-file>` 而不是 `git add -A`。
- commit message 描述仓库发生了什么，不要写"为了测试"、"修复了一个问题"等模糊描述。

### 文档规则

- 每次改代码后，在下一个 commit 更新相关文档。
- 改了源码 → 更新 `implementation.md`（改动文件表格）。
- 改了运行方式 → 更新 `README.md`（快速运行部分）。
- 新发现或决策 → 记录到 `README.md` 的"当前状态"。

### 调试规则

- 遇到报错，先查 `implementation.md` 的"注意事项"。
- 不确定的参数，先查 `miles/utils/arguments.py`（grep 参数名）。
- 框架内部行为，先 grep 再猜。

## 关键设计决策（已定，不要推翻）

| 决策 | 原因 |
|------|------|
| 用 `match` 模式而非独立脚本 | 走真实完整 pipeline，能发现生产路径中的 bug |
| 不用 `--get-mismatch-metrics` | `use_rollout_logprobs` 未设置，Megatron 无条件重算，该 flag 多余且触发 assert |
| `--skip-train-step` 而非 `--forward-only` | `forward-only` 是别的模式，`skip-train-step` 是我们加的新 flag |
| 用 `rollout-seed=42` + 关闭 `rollout-shuffle` | 跨次运行结果可复现 |
| `load_function` 支持文件路径格式 | `tools/true-on-policy/` 含连字符，不能作为 Python 模块名 |

## 当前已知问题

无（上次运行的 `load_function` 报错已修复）。如有新问题，记录在 `README.md` 的"当前状态"里。
