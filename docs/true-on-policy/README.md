---
title: True-On-Policy 一致性测试
---

# True-On-Policy 一致性测试

## 目标

验证在 `--true-on-policy-mode` 下，SGLang 推理引擎记录的 `rollout_log_probs` 与 Megatron 训练引擎对同一 token 序列重算的 `log_probs` 数值完全一致。

判定标准：Pearson r = 1.0，MSE = 0。

## 当前状态

已实现 `match` 测试模式，工具就位，可直接运行。

上次运行发现 `load_function` 不支持文件路径格式，已修复（`miles/utils/misc.py`）。

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
python tools/true-on-policy/run_megatron.py --cuda-visible-devices 5 --skip-prepare

# 5. 分析结果
python tools/true-on-policy/compute_metrics.py --save-dir /tmp/true-on-policy
```

输出自动保存到 `log/train_YYYYMMDD_HHMMSS.txt`。

## 预期输出

```
=== Logprob consistency ===
  tokens:        262144
  Pearson r:     1.000000
  MSE:           0.000000e+00
  mean |diff|:   0.000000e+00
```

偏差 → logprob 不一致，用 per-layer hidden state 定位（见 implementation.md）。

## 文件索引

- `workflow.md` — AI 工作指南：从零上手、工作规则、已定决策
- `implementation.md` — 源码改动、工具说明、参数参考
- `code-map.md` — 相关代码位置
