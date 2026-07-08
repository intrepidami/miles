---
title: True-On-Policy 相关代码位置
---

# 相关代码位置

## 改动文件

| 文件 | 作用 |
|------|------|
| `miles/utils/arguments.py` | CLI 参数注册，包含 `--skip-train-step`、`--true-on-policy-mode` 等 |
| `miles/backends/megatron_utils/actor.py` | Megatron actor 主循环，logprob 重算和 train 调用在此 |
| `miles/backends/training_utils/log_utils.py` | `_maybe_save_logprobs()` 落盘逻辑 |
| `miles/utils/misc.py` | `load_function()`，支持 `/path/file.py:func` 格式 |
| `scripts/run_qwen3_4b.py` | Qwen3 启动脚本，含 `match` 模式 |

## True-On-Policy 框架

| 文件 | 作用 |
|------|------|
| `miles/true_on_policy/config.py` | `build_true_on_policy_launch_plan()`，生成确定性参数和 env vars |
| `miles/true_on_policy/model_profiles.py` | 各模型的 true-on-policy 支持配置 |

## 损失计算

| 文件 | 作用 |
|------|------|
| `miles/backends/training_utils/loss_hub/losses.py` | logprob mismatch 判定，`custom_tis_function_path` 调用点 |
| `miles/backends/training_utils/loss_hub/corrections.py` | `vanilla_tis_function` 等 IS 权重函数 |

## 工具文件

| 文件 | 作用 |
|------|------|
| `tools/true-on-policy/run_megatron.py` | 测试启动器 |
| `tools/true-on-policy/megatron_hs_hook.py` | Megatron hidden state 捕获 |
| `tools/true-on-policy/compute_metrics.py` | 离线指标计算 |
