---
title: True-On-Policy 一致性测试
---
# True-On-Policy 一致性测试

## 目标

验证在 `--true-on-policy-mode` 下，SGLang 推理引擎记录的 `rollout_log_probs` 与 Megatron 训练引擎对同一 token 序列重算的 `log_probs` 数值完全一致。

判定标准：MSE = 0，mean |diff| = 0（Pearson r = 1.0 仅作 sanity check）。

注意：Pearson 对失配不敏感——`r ≈ 1 − MSE/(2·Var)`，基线（不开 true-on-policy）r 就已 ≈0.999+，开关前后差别只在小数点后四五位。判别效果看 MSE / mean|diff| / max|diff|，详见 `implementation.md` 的"为什么 Pearson 对开关不敏感"。

## 当前状态

**当前目标（2026-07-09 起）**：`run_match.py` 支持 FSDP backend + 2 GPU（DP=2），跑通 true-on-policy，对齐 `examples/true_on_policy/run_simple.py` 的 FSDP 配置。支持改动已完成（见 `implementation.md`"FSDP match 支持"），管线已跑通（2026-07-09，单卡 fsdp + true-on-policy 全流程到指标汇总）。

**重要前提校准**：`run_simple.py` / 官方 README 宣称的 `train_rollout_logprob_abs_diff` 严格为 0 是 **~2025-11（约 8 个月前，SGLang v0.5.10 时代）**验证的。当前 H800 上的 SGLang 明显更新（模块路径迁移：`fused_moe_triton` 位置变了；weight sync 需要 `begin_weight_update` 会话）。**"diff=0"的结论不能直接假设在当前栈成立**。2026-07-09 首次 fsdp true-on-policy 实测：两侧 logprob 均值差 7.4e-4（`log_probs=-0.60538` vs `rollout_log_probs=-0.60464`）——待 `compute_metrics.py` 出逐 token 指标后判断是版本漂移还是配置问题（先查 SGLang 启动 log 中 fa3/deterministic 是否静默回退）。

**复现 v0.5.10 时代镜像（2026-07-09 定案，未执行）**：要在当前 H800 上复现官方 diff=0 的原始栈（sglang v0.5.10 + torch 2.9.1 + cu129），不能在 `radixark/miles:latest` 上原地降级——现镜像 torch 2.11，全降会砸烂按 2.11 编译的 flash-attn/TE/apex。正确做法：用 miles 仓库 v0.5.10 时代的 Dockerfile 从 `lmsysorg/sglang:v0.5.10` 底座重建。坐标（均已验证存在）：

- 时代 miles commit `bbea8556101011b23d61dd13ec78aa9511d63508`（2026-05-22 升 v0.5.12 前最后一版；Dockerfile 默认 `SGLANG_IMAGE_TAG=v0.5.10`、`WHEELS_TAG=cu129-x86_64`）。构建上下文必须整体 checkout 到该 commit（requirements.txt 要同代）。
- 官方 sglang `v0.5.10` tag commit `1519acf37c23f2189adb93f57ca9cd2db1bebf18`。**必须显式传 `--build-arg SGLANG_COMMIT=<该 SHA>`**：不传则 Dockerfile checkout `sglang-miles` 分支最新 HEAD，sglang 又被升上去。
- wheels release `cu129-x86_64`（yueming-yuan/miles-wheels，2026-04-17 发布）：时代 flash-attn/FA3/apex/sgl-router，按 torch 2.9.1 + cu129 编译。

```bash
git clone https://github.com/radixark/miles.git miles-v0510-build && cd miles-v0510-build
git checkout bbea8556101011b23d61dd13ec78aa9511d63508
docker build -f docker/Dockerfile \
  --build-arg SGLANG_COMMIT=1519acf37c23f2189adb93f57ca9cd2db1bebf18 \
  -t miles:sglang-v0.5.10-torch291 .
docker run --rm miles:sglang-v0.5.10-torch291 \
  python -c "import torch, sglang; print(torch.__version__, sglang.__version__)"  # 2.9.1 0.5.10
```

已知坑：时代 Dockerfile 有一行未钉版本的 `pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/`，nightly 索引变动可能挂——挂了加 `==0.1.8`（现 main 钉的版本）。镜像烤进的 miles 是当前 main，按快速运行流程 `pip install -e .` 自己的工作树覆盖即可。dev 分支已适配新 sglang 的点（weight sync `begin_weight_update`、`fused_moe_triton` tolerant import）在 0.5.10 下需向后兼容，跑挂先查 weight sync 调用处。

**已搁置：Megatron hidden states shape 对齐 / kernel 交换定位任务**。搁置时的进度：

- token 对齐逐层对比已实现（`--micro-batch-size 1` 采集 + cosine 列匹配），代码在 `compute_metrics.py`，原理见 `implementation.md`"对齐的实现原理"。
- 2026-07-08 实测（Qwen3-0.6B 单卡，262144 tokens）：true-on-policy MSE=1.204e-3 vs baseline MSE=1.325e-3——只降 9%，**Megatron kernel 交换未生效**。已排除 flag 下发问题与 Megatron patch 缺失；数值指纹（max/p99 为二进制格点、mean|diff|≈bf16 1–2 ULP）指向双侧 bf16 路径生效但 kernel 结果不同。证据链与嫌疑清单（attention backend 不一致 / SGLang 静默回退 / 覆盖不含 final norm+lm_head）见 `implementation.md`"Qwen3-0.6B true-on-policy 效果不明显：原因分析"。
- 2026-07-09 run `20260709_025156`（baseline）：修了 thd padding 裁剪、CUDA graph 吞 decode dump（match 模式已加 `--sglang-disable-cuda-graph`）；该次 unaligned 对比无参考价值。
- **恢复点**：跑一次 true-on-policy + `--dumper-enable` 的 match，用 token 对齐逐层对比定位分歧起始层（layer 0 即分歧 → attention/embedding 入口；逐层递增 → kernel 累积误差；各层皆小但 logprob 差大 → final norm/lm_head）。

工具变更（2026-07-09）：

- `run_megatron.py` 改名 `run_match.py`；hidden-state hook 与 tensor dumper 落盘默认**关闭**，需显式 `--capture-hidden-states` / `--dumper-enable`。
- `compute_metrics.py` 的 hidden state 对比默认**不计算**，需显式 `--hidden-states`。

实测发现开/关 true-on-policy 对 Pearson r 影响极小——这是 Pearson 的灵敏度问题（见 `implementation.md`"为什么 Pearson 对开关不敏感"）；单卡 Qwen3-0.6B 下各开关的实际生效面见 `implementation.md`"true-on-policy 在 Qwen3-0.6B 单卡 match 下的生效面"。

## 快速运行

```bash
# 1. 同步代码
git pull

# 2. 安装（首次）
pip install -e . --no-build-isolation --no-deps
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

# 3. 编辑路径（MODEL_DIR、DATA_DIR、MEGATRON_PATH、BASE_DIR）
vim tools/true-on-policy/run_match.py

# 4a. Megatron 单卡运行（--skip-prepare 跳过模型下载和 checkpoint 转换）
python tools/true-on-policy/run_match.py --cuda-visible-devices 5 --num-gpus-per-node 1 --num-nodes 1 --skip-prepare --train-backend megatron --true-on-policy

# 4b. FSDP 2 卡运行（DP=2，当前目标配置）
python tools/true-on-policy/run_match.py --cuda-visible-devices 4,5 --num-gpus-per-node 2 --num-nodes 1 --skip-prepare --train-backend fsdp --true-on-policy

# 5. 分析结果（--hidden-states 可选：额外做逐层 hidden state 对比）
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/<YYYYMMDD_HHMMSS>
```

输出自动保存到运行目录 `log/log_YYYYMMDD_HHMMSS.txt`。

可选参数

- `--cuda-visible-devices 1,2,3,4`
- `--num-gpus-per-node 2`
- `--num-nodes 1`
- `--skip-prepare`
- `--train-backend {megatron,fsdp}`: Training backend. Defaults to `megatron`.
- `--true-on-policy`: Enable true-on-policy mode. If omitted, true-on-policy is disabled.
- `--megatron-tp N` / `--megatron-pp N` / `--megatron-cp N` / `--megatron-dp N`: 覆盖 megatron 并行度（默认模型推导：0.6B 为 TP=1/CP=1，PP=1，DP=1）。actor 占 TP×PP×CP×DP 卡，须 ≤ `--num-gpus-per-node`。例 TP=2/CP=2/DP=2 共 8 卡：`--num-gpus-per-node 8 --megatron-tp 2 --megatron-cp 2 --megatron-dp 2`。注意：TP>1 自动开 sequence-parallel，CP>1 用 a2a——两者都改 kernel 数值路径，与单卡/SGLang(TP=1) 的 logprob 可比性口径不同。fsdp backend 忽略这四个参数。推导原理见 `implementation.md`"并行度与批量推导"。
- `--capture-hidden-states`: 开启 Megatron hidden-state hook 落盘（`megatron_hs/`，默认关；fsdp backend 下忽略）
- `--dumper-enable`: 开启 SGLang/Megatron tensor dumper 落盘（`tensor_cmp/`，默认关）

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
