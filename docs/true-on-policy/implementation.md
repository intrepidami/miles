---
title: True-On-Policy 实现细节
---

# 实现细节

## 源码改动

| 文件 | 改动 |
|------|------|
| `miles/utils/arguments.py` | 新增 `--skip-train-step`：跳过 backward + optimizer，rollout 和 log-prob 前向仍正常执行 |
| `miles/backends/megatron_utils/actor.py` | `train()` 调用受 `args.skip_train_step` 保护 |
| `miles/backends/training_utils/log_utils.py` | 新增 `_maybe_save_logprobs()`：在 `log_rollout_data()` 入口，当 `MILES_TRUE_ON_POLICY_SAVE_DIR` 设置时落盘 logprobs |
| `miles/utils/misc.py` | `load_function` 支持文件路径格式 `/path/file.py:func`（原来只支持 dot-notation） |
| `scripts/run_qwen3_4b.py` | 新增 `match` 模式 |

## match 模式参数

`mode="match"` 时 `scripts/run_qwen3_4b.py` 使用以下配置：

| 参数 | 值 | 原因 |
|------|----|------|
| `rollout-batch-size` | 128 | 足够样本量 |
| `n-samples-per-prompt` | 1 | - |
| `rollout-max-response-len` | 2048 | - |
| `num-rollout` | 1 | 单步验证 |
| `global-batch-size` | 128 | - |
| `rollout-shuffle` | 关闭 | 保证可复现 |
| `rollout-seed` | 42（默认） | 确定性采样 |
| `--skip-train-step` | 开启 | 不修改权重，可反复跑 |
| `--true-on-policy-mode` | 开启 | 触发 Megatron 重算 logprobs |

注：`--true-on-policy-mode` 本身不设置 `--use-rollout-logprobs`，所以 `actor.py` 中 `not args.use_rollout_logprobs` 已为 True，Megatron 无条件重算 logprobs，无需 `--get-mismatch-metrics`。

## 工具

### `tools/true-on-policy/run_megatron.py`

测试启动器，封装 `scripts/run_qwen3_4b.py` 的 `match` 模式。

顶部 CONFIG 块配置路径：

```python
MODEL_NAME        = "Qwen3-0.6B"
MODEL_DIR         = "/root/models"
DATA_DIR          = "/root/datasets"
OUTPUT_DIR        = "/root/output"
MEGATRON_PATH     = "/root/Megatron-LM"
SAVE_DIR          = "/root/code/true-on-policy"
CAPTURE_HIDDEN_STATES = True
```

CLI 参数：

```
--skip-prepare          跳过模型下载和 checkpoint 转换
--cuda-visible-devices  CUDA_VISIBLE_DEVICES（如 "5" 或 "0,1,2,3"）
--num-gpus-per-node     覆盖 GPU 数（默认从 hardware 推导）
--num-nodes             覆盖节点数（默认 1）
```

### `tools/true-on-policy/megatron_hs_hook.py`

通过 `--custom-megatron-before-log-prob-hook-path /path/megatron_hs_hook.py:register` 注入。

在 log-prob 前向时捕获每层 hidden states（检测 `self_attention + mlp` 属性）。用 `atexit` 在进程退出时写文件。

输出 `$SAVE_DIR/megatron_hs/rank_0/layer_NNN.npy`，shape `[total_tokens, hidden_dim]`，float32。

Megatron hidden state 原始布局为 `[seq, batch, hidden]`，hook 转置为 `[total_tokens, hidden]`。

### `tools/true-on-policy/compute_metrics.py`

```bash
python tools/true-on-policy/compute_metrics.py --save-dir /root/code/true-on-policy
python tools/true-on-policy/compute_metrics.py --save-dir /root/code/true-on-policy --rollout 0
python tools/true-on-policy/compute_metrics.py --save-dir /root/code/true-on-policy --json
```

输出：
- Pearson r（logprobs 全局相关，理论值 1.0）
- MSE、mean/max/p99 |diff|
- per-layer mean L2 norm（有 hidden states 时）
- Cumulative MSE

## 落盘文件结构

```
$SAVE_DIR/
  rollout_0000/
    log_probs.npy           # [total_tokens] float32 — Megatron 重算
    rollout_log_probs.npy   # [total_tokens] float32 — SGLang 原始
  rollout_0001/
    ...
  megatron_hs/
    rank_0/
      layer_000.npy         # [total_tokens, hidden_dim] float32
      layer_001.npy
      ...
```

只有 TP rank=0、PP last stage 写文件。多 rank 环境下其他 rank 不产生文件，属正常。

## 环境变量

| 变量 | 设置方 | 效果 |
|------|--------|------|
| `MILES_TRUE_ON_POLICY_SAVE_DIR` | `run_megatron.py` 自动设置 | 触发 logprob + hidden state 落盘 |
