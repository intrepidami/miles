---
title: True-On-Policy 一致性测试：实现说明
description: 相比 main 分支，dev 分支实现了什么、如何实现、有什么用。
---

# True-On-Policy 一致性测试：实现说明

## 背景

Miles 的 true-on-policy 模式要求：SGLang 推理引擎生成 token 时计算的 `rollout_log_probs`，必须与 Megatron 训练引擎对同一批 token 重算得到的 `log_probs` 完全一致。任何偏差都意味着训练用了错误的 logprobs，梯度计算不可信。

本 dev 分支在不破坏原有训练流程的前提下，实现了一套最小化的前向一致性测试工具。

---

## 相比 main 的改动

### 1. 新增 `--skip-train-step` 参数

**文件**：`miles/utils/arguments.py`、`miles/backends/megatron_utils/actor.py`

```
--skip-train-step
```

跳过 actor 的反向传播和 optimizer 更新，但 rollout 和 log-prob 前向（`compute_log_prob` + `log_rollout_data`）仍正常执行。

用途：隔离前向计算，不污染模型权重，单次运行即可拿到双引擎 logprobs 用于对比。

### 2. logprobs 自动落盘

**文件**：`miles/backends/training_utils/log_utils.py`

新增 `_maybe_save_logprobs()`，在 `log_rollout_data()` 入口处调用。

当环境变量 `MILES_TRUE_ON_POLICY_SAVE_DIR` 被设置时，每个 rollout step 自动在该目录下保存：

```
rollout_0000/
  log_probs.npy          # Megatron 重算 logprobs，shape [total_tokens]，float32
  rollout_log_probs.npy  # SGLang 原始 logprobs，shape [total_tokens]，float32
```

只在 TP rank=0、PP last stage 上写文件，避免重复写。

### 3. `match` 模式

**文件**：`scripts/run_qwen3_4b.py`

在 `ScriptArgs.mode` 中新增 `"match"` 选项，配置如下：

| 参数 | 值 |
|------|----|
| `rollout-batch-size` | 128 |
| `n-samples-per-prompt` | 1 |
| `rollout-max-response-len` | 2048 |
| `num-rollout` | 2 |
| `global-batch-size` | 128 |
| `rollout-shuffle` | 关闭（保证可复现） |
| `rollout-seed` | 42（默认，保证同一 seed 下生成确定） |
| `--skip-train-step` | 开启 |
| `--true-on-policy-mode` | 开启 |

共测试 2 步 rollout，每步 128 条 prompt × 最多 2048 token。

---

## 新增工具

### `tools/true-on-policy/run_megatron.py`

封装了完整的测试启动流程：

1. 下载 Qwen3-0.6B（或其他模型）
2. 下载 dapo-math-17k 数据集
3. 转换 HF checkpoint 为 Megatron 格式
4. 以 `match` 模式启动训练流程

关键特性：
- CONFIG 块统一管理路径（`MODEL_DIR`、`DATA_DIR`、`MEGATRON_PATH` 等）
- `--cuda-visible-devices`、`--num-gpus-per-node`、`--num-nodes` 支持灵活的 GPU 配置
- 输出自动 tee 到 `log/train_YYYYMMDD_HHMMSS.txt`
- `--skip-prepare` 跳过下载和转换（复用已有 checkpoint）

### `tools/true-on-policy/megatron_hs_hook.py`

通过 `--custom-megatron-before-log-prob-hook-path` 注入，在 Megatron log-prob 前向时捕获每层 hidden states。

检测方式：检查 `hasattr(module, 'self_attention') and hasattr(module, 'mlp')` 标识 transformer layer。

输出：

```
megatron_hs/rank_0/layer_000.npy   # shape [total_tokens, hidden_dim]，float32
megatron_hs/rank_0/layer_001.npy
...
```

用 `atexit` 在进程退出时统一写文件（hook 在前向中积累，退出时保存）。

### `tools/true-on-policy/compute_metrics.py`

离线分析工具，读取 `.npy` 文件，计算：

| 指标 | 含义 |
|------|------|
| Pearson r | logprobs 全局相关系数，理论值 1.0 |
| MSE | 均方误差，理论值 0 |
| mean/max/p99 \|diff\| | 绝对误差分布 |
| per-layer mean L2 norm | 每层 hidden state 范数 |
| Cumulative MSE | 各层 L2 范数平方的均值 |

```bash
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy --json
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy --rollout 0
```

---

## 有什么用

### 验证 true-on-policy 实现正确性

Pearson r ≈ 1.0、MSE ≈ 0 → SGLang 和 Megatron 在同一 token 序列上数值一致 → true-on-policy 可信。

任何偏差（r < 1 或 MSE > 0）→ 定位 bug：权重同步错误、精度格式不一致、kernel 行为差异等。

### 隔离调试，无副作用

`--skip-train-step` 使单次运行不修改权重，可以反复跑同一 checkpoint 对比不同配置的输出，不需要担心权重漂移干扰结果。

### per-layer hidden state 辅助定位

logprobs 不一致时，逐层对比 hidden state 可以精确定位是哪一层开始出现数值分叉，缩小排查范围。

### 真实数据，真实路径

使用 dapo-math-17k 真实 prompt，走完整 Ray + SGLang + Megatron 推理训练流程，不是合成数据或简化路径，能发现生产流程中的实际问题。

---

## 快速启动

```bash
# 安装
pip install -e . --no-build-isolation --no-deps
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

# 编辑 CONFIG（路径等）
vim tools/true-on-policy/run_megatron.py

# 首次运行（含下载和 checkpoint 转换）
python tools/true-on-policy/run_megatron.py --cuda-visible-devices 5

# 后续运行（跳过下载）
python tools/true-on-policy/run_megatron.py --cuda-visible-devices 5 --skip-prepare

# 分析结果
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy
```

---

## 改动文件汇总

| 文件 | 类型 | 改动 |
|------|------|------|
| `miles/utils/arguments.py` | 框架 | 新增 `--skip-train-step` |
| `miles/backends/megatron_utils/actor.py` | 框架 | `train()` 受 `skip_train_step` 保护 |
| `miles/backends/training_utils/log_utils.py` | 框架 | `_maybe_save_logprobs()` 自动落盘 |
| `miles/utils/misc.py` | 框架 | `load_function` 支持文件路径格式 `/path/file.py:func` |
| `scripts/run_qwen3_4b.py` | 脚本 | 新增 `match` 模式 |
| `tools/true-on-policy/run_megatron.py` | 工具 | 测试启动器 |
| `tools/true-on-policy/megatron_hs_hook.py` | 工具 | Megatron hidden state 捕获 hook |
| `tools/true-on-policy/compute_metrics.py` | 工具 | 离线指标计算 |
| `tools/true-on-policy/README.md` | 文档 | 工具使用说明（英文） |
| `docs/true-on-policy/` | 文档 | 工作日志、规则、代码地图等 |
