---
title: True-On-Policy 一致性测试
---
# True-On-Policy 一致性测试

## 目标

验证在 `--true-on-policy-mode` 下，SGLang 推理引擎记录的 `rollout_log_probs` 与 Megatron 训练引擎对同一 token 序列重算的 `log_probs` 数值完全一致。

判定标准：MSE = 0，mean |diff| = 0（Pearson r = 1.0 仅作 sanity check）。

注意：Pearson 对失配不敏感——`r ≈ 1 − MSE/(2·Var)`，基线（不开 true-on-policy）r 就已 ≈0.999+，开关前后差别只在小数点后四五位。判别效果看 MSE / mean|diff| / max|diff|，详见 `implementation.md` 的"为什么 Pearson 对开关不敏感"。

## 当前状态

已实现 `match` 测试模式，工具就位，可直接运行。

**注意：当前 `run_megatron.py` 设置 `true_on_policy=False`**（不加 `--true-on-policy-mode` 和 `--recompute-logprobs-via-prefill`），跑的是**基线失配测量**：`rollout_log_probs` 是 SGLang decode 时的值，Megatron 用标准 kernel 重算。此配置下 Pearson r = 1.0 / MSE = 0 的判定标准**不成立**，仅在开启 true-on-policy 后成立。

上次运行发现 `load_function` 不支持文件路径格式，已修复（`miles/utils/misc.py`）。

实测发现开/关 true-on-policy 对 Pearson r 影响极小——这是 Pearson 的灵敏度问题（见 `implementation.md`"为什么 Pearson 对开关不敏感"）；单卡 Qwen3-0.6B 下各开关的实际生效面见 `implementation.md`"true-on-policy 在 Qwen3-0.6B 单卡 match 下的生效面"。

**2026-07-08 实测（Qwen3-0.6B 单卡，262144 tokens）**：true-on-policy MSE=1.204e-3 vs baseline MSE=1.325e-3——只降 9%，**kernel 对齐未生效**。已排除 flag 下发问题（EXEC 命令行含全部开关）与 Megatron patch 缺失（`use_true_on_policy_backend` 在位）；数值指纹（max/p99 为二进制格点、mean|diff|≈bf16 1–2 ULP）指向双侧 bf16 路径生效但 kernel 结果不同。完整证据链与嫌疑清单见 `implementation.md`"Qwen3-0.6B true-on-policy 效果不明显：原因分析"。下一步：`20260709_025156`（micro-batch-size 1 重跑）跑 token 对齐逐层对比定位分歧层。

## 快速运行

```bash
# 1. 同步代码
git pull

# 2. 安装（首次）
pip install -e . --no-build-isolation --no-deps
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

# 3. 编辑路径（MODEL_DIR、DATA_DIR、MEGATRON_PATH、SAVE_DIR）
vim tools/true-on-policy/run_megatron.py

# 4. 运行（--skip-prepare 跳过模型下载和 checkpoint 转换）
python tools/true-on-policy/run_megatron.py --cuda-visible-devices 5 --num-gpu-per-node 1 --num-nodes 1 --skip-prepare --train-backend megatron --true-on-policy

# 5. 分析结果
python tools/true-on-policy/compute_metrics.py --save-dir /root/code/true-on-policy
```

输出自动保存到 `log/train_YYYYMMDD_HHMMSS.txt`。

可选参数

- `--cuda-visible-devices 1,2,3,4`
- `--num-gpu-per-node 1`
- `--num-nodes 1`
- `--skip-prepare`
- `--train-backend {megatron,fsdp}`: Training backend. Defaults to `megatron`.
- `--true-on-policy`: Enable true-on-policy mode. If omitted, true-on-policy is disabled.
- `--capture-hidden-states`
- `--dumper-enable`

## 预期输出

```
=== Logprob match ===
  tokens:        262144
  Pearson r:     1.000000
  MSE:           0.000000e+00
  mean |diff|:   0.000000e+00
```

结果写入 `<save_dir>/metrics/match.csv`（每次覆盖）。

偏差 → logprob 不一致，用 per-layer hidden state 定位（见 implementation.md）。

## 文件索引

- `workflow.md` — AI 工作指南：从零上手、工作规则、已定决策
- `implementation.md` — 源码改动、工具说明、参数参考
- `code-map.md` — 相关代码位置
