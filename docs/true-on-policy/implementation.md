---
title: True-On-Policy 实现细节
---
# 实现细节

## 源码改动

| 文件                                           | 改动                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `miles/utils/arguments.py`                   | 新增`--skip-train-step`：跳过 backward + optimizer，rollout 和 log-prob 前向仍正常执行                                 |
| `miles/backends/megatron_utils/actor.py`     | `train()` 调用受 `args.skip_train_step` 保护                                                                         |
| `miles/backends/training_utils/log_utils.py` | 新增`_maybe_save_logprobs()`：在 `log_rollout_data()` 入口，当 `MILES_TRUE_ON_POLICY_SAVE_DIR` 设置时落盘 logprobs |
| `miles/utils/misc.py`                        | `load_function` 支持文件路径格式 `/path/file.py:func`（原来只支持 dot-notation）                                     |
| `scripts/run_qwen3_4b.py`                    | 新增`match` 模式                                                                                                       |

## match 模式参数

`mode="match"` 时 `scripts/run_qwen3_4b.py` 使用以下配置：

| 参数                         | 值         | 原因                        |
| ---------------------------- | ---------- | --------------------------- |
| `rollout-batch-size`       | 128        | 足够样本量                  |
| `n-samples-per-prompt`     | 1          | -                           |
| `rollout-max-response-len` | 2048       | -                           |
| `num-rollout`              | 1          | 单步验证                    |
| `global-batch-size`        | 128        | -                           |
| `rollout-shuffle`          | 关闭       | 保证可复现                  |
| `rollout-seed`             | 42（默认） | 确定性采样                  |
| `--skip-train-step`        | 开启       | 不修改权重，可反复跑        |
| `--true-on-policy-mode`    | 开启       | 触发 Megatron 重算 logprobs |

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
SAVE_DIR          = "/root/true-on-policy"
CAPTURE_HIDDEN_STATES = True
DUMPER_ENABLE     = True        # SGLang dumper：捕获 rollout + log-prob pass 所有张量
DUMPER_DIR        = "/root/true-on-policy/tensor_cmp"
```

`DUMPER_ENABLE=True` 时自动附加：

- `--dumper-enable --dumper-dir $DUMPER_DIR`
- `--dumper-fwd-only enable=true non_intrusive_mode=all filter='<hidden_state_filter>'`
- `--dumper-inference enable=true non_intrusive_mode=all filter='<hidden_state_filter>'`

filter 表达式（Python eval 对 tags dict）：

```python
"layer_id is not None and name is not None and name.endswith('.mlp.output')"
```

只捕每个 transformer 层的 `mlp.output`，即 MLP 子层输出，是每层 hidden state 最好的代理。`layers.N.output` 本身是 tuple 不直接落盘；`"output" in name` 过于宽泛会捕到 `q_norm.output`、`v_proj.output` 等中间张量。

tags 中 `layer_id` 由 SGLang dumper 对匹配 `layers.\d+` 的模块自动注入，`name` 形如 `non_intrusive__model.layers.0.mlp.output`。

输出目录结构：

```
/root/true-on-policy/tensor_cmp/
  fwd_only/      # Megatron log-prob pass 每层 hidden states（重算时）
  engines/
    engine_0/    # SGLang inference 每层 hidden states（生成时）
```

两者捕获的不是同一个东西：SGLang 用 KV cache 自回归生成，Megatron 做完整序列前向重算。对比两者是 true-on-policy 调试的目的。

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
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy --rollout 0
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy --json
```

输出：

- `=== Logprob match ===`：Pearson r（理论值 1.0）、MSE、mean/max/p99 |diff|
- per-layer mean L2 norm（有 hidden states 时）、Cumulative MSE
- 结果自动追加到 `<save_dir>/metrics/match.csv`（首次运行写 header）

## 落盘文件结构与数据来源

### `/root/true-on-policy/`（SAVE_DIR）

```
rollout_0000/
  log_probs.npy           # [total_tokens] float32 — Megatron 重算 logprobs
  rollout_log_probs.npy   # [total_tokens] float32 — SGLang 生成时 logprobs
megatron_hs/
  rank_0/
    layer_000.npy         # [total_tokens, hidden_dim] float32
    layer_001.npy
    ...
```

| 文件 | 生产者 | 调用链 |
|------|--------|--------|
| `rollout_*/log_probs.npy` | `log_utils.py:_maybe_save_logprobs()` | `actor.py:train()` → `log_rollout_data()` → `_maybe_save_logprobs()`，由 `MILES_TRUE_ON_POLICY_SAVE_DIR` 触发 |
| `rollout_*/rollout_log_probs.npy` | 同上 | 同上 |
| `megatron_hs/rank_0/layer_NNN.npy` | `megatron_hs_hook.py:register()` | `model.py:compute_log_probs()` 前调用，`atexit` 写盘，由 `--custom-megatron-before-log-prob-hook-path` 注入 |

只有 TP rank=0、PP last stage 写文件。

### `/root/output/{run_id}/`（OUTPUT_DIR）

```
checkpoints/              # Megatron 模型权重（match 模式权重不变）
dump_details/
  rollout_data/{rollout_id}.pt   # {rollout_id, samples: [Sample.to_dict()]}
  train_data/{rollout_id}_{rank}.pt  # {rollout_id, rank, rollout_data: RolloutBatch}
```

| 文件 | 生产者 | 调用链 |
|------|--------|--------|
| `rollout_data/*.pt` | `debug_data.py:save_debug_rollout_data()` | `rollout_manager.py:rollout()` 完成后，含每条 Sample 的 `rollout_log_probs`、tokens、rewards 等 |
| `train_data/*.pt` | `train_dump_utils.py:save_debug_train_data()` | `actor.py:train()` 末尾，含完整 `RolloutBatch`（同时有 `log_probs` 和 `rollout_log_probs`），可直接对比两者 |

### `/root/true-on-policy/tensor_cmp/`（DUMPER_DIR）

```
fwd_only/                 # Megatron log-prob pass 每层 hidden states（重算时）
engines/engine_0/         # SGLang inference 每层 hidden states（生成时）
```

| 目录 | 生产者 | 调用链 |
|------|--------|--------|
| `fwd_only/` | `dumper_utils.py:DumperMegatronUtil(FWD_ONLY)` | `model.py:compute_log_probs()` 中 `DumperMegatronUtil` 注入 PyTorch forward hook，`--dumper-fwd-only enable=true` 触发 |
| `engines/engine_0/` | SGLang 内置 `sglang.srt.debug_utils.dumper` | `server_group.py` 启动时注入 `DUMPER_SERVER_PORT` 环境变量，`sglang_rollout.py:configure_sglang()` 通过 HTTP `/dumper/configure` 激活，`--dumper-inference enable=true` 触发 |

## 环境变量

| 变量                              | 设置方                       | 效果                             |
| --------------------------------- | ---------------------------- | -------------------------------- |
| `MILES_TRUE_ON_POLICY_SAVE_DIR` | `run_megatron.py` 自动设置 | 触发 logprob + hidden state 落盘 |
