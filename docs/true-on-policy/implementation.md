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

注：`--true-on-policy-mode` **不再**是 match 模式的默认参数；它由 `run_megatron.py` 在 `extra_args` 中显式附加，使 `scripts/run_qwen3_4b.py match` 模式本身更通用。`--true-on-policy-mode` 本身不设置 `--use-rollout-logprobs`，所以 `actor.py` 中 `not args.use_rollout_logprobs` 已为 True，Megatron 无条件重算 logprobs，无需 `--get-mismatch-metrics`。

`build_launch_plan`（`miles/true_on_policy/config.py`）自动附加 `--recompute-logprobs-via-prefill`，rollout 结束后 SGLang 对完整序列做一次 prefill 重算，覆盖 decode 时的 logprobs。

## 在线 Metric：train_rollout_logprob_abs_diff

`losses.py:policy_loss_function` 在每次训练 forward 中计算（`rollout_log_probs` 存在时）：

| 变量 | 来源 |
|------|------|
| `train_scored_log_probs` | `batch["log_probs"]`（Megatron 完整序列重算） |
| `rollout_log_probs` | `batch["rollout_log_probs"]`（SGLang prefill 重算，由 `--recompute-logprobs-via-prefill` 触发） |

```python
abs_diff = (train_scored_log_probs - rollout_log_probs).abs()
# 仅统计 active_tokens，nan/inf 归零
train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)

# KL(rollout ‖ train)，Schulman k3 近似，per-token clamp [-10, 10]
train_rollout_kl = sum_of_sample_mean(
    compute_approx_kl(rollout_log_probs, train_scored_log_probs, kl_loss_type="low_var_kl")
)
```

两个 metric 写入 `reported_loss` 并上报 WandB/tensorboard：
- `train_rollout_logprob_abs_diff`：逐 token 绝对差的 per-sample 均值
- `train_rollout_kl`：KL(SGLang ‖ Megatron) 的 per-sample 均值

这是训练时的在线指标。`compute_metrics.py` 做的是相同对比的离线版（从落盘 `.npy` 文件计算）。

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
BASE_DIR          = "/root/true-on-policy"   # 每次运行在 BASE_DIR/{YYYYMMDD_HHMMSS}/ 下存数据
CAPTURE_HIDDEN_STATES = True
DUMPER_ENABLE     = True        # SGLang dumper：捕获 rollout + log-prob pass 张量
```

`DUMPER_DIR` 由运行时自动推导为 `{BASE_DIR}/{timestamp}/tensor_cmp`，无需手动配置。

`DUMPER_ENABLE=True` 时自动附加：

- `--dumper-enable --dumper-dir {run_dir}/tensor_cmp`
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

支持两种数据来源（logprob 输入互斥，CSV 始终写 `<save_dir>/metrics/match.csv`）。

**参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `--save-dir PATH` | 必填 | 运行目录（`BASE_DIR/YYYYMMDD_HHMMSS/`）；Mode 1 读 `.npy`，始终写 CSV |
| `--dump-details PATH` | 可选 | Mode 2：`dump_details/` 目录，从 `train_data/*.pt` 读 logprobs |
| `--rank INT` | 可选，默认 0 | Mode 2：读 `{rollout_id}_{rank}.pt` 中哪个 rank |
| `--rollout INT...` | 可选，默认全部 | 指定 rollout ID，如 `--rollout 0 1` |
| `--model-name STR` | 可选 | 写入 CSV 的模型名（如 `Qwen3-0.6B`） |
| `--batch-size INT` | 可选 | rollout-batch-size，写入 CSV |
| `--max-response-len INT` | 可选 | rollout-max-response-len，写入 CSV |
| `--json` | flag | 额外打印 JSON |

**Mode 1 — .npy（默认）**

```bash
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022 --rollout 0
```

读取文件：

| 文件 | 作用 |
|------|------|
| `rollout_*/log_probs.npy` | Megatron 重算 logprobs，`[total_tokens]` float32 |
| `rollout_*/rollout_log_probs.npy` | SGLang 生成时 logprobs，`[total_tokens]` float32 |
| `tensor_cmp/fwd_only/*.pt` | Megatron log-prob pass 每层 mlp.output（dumper 产生） |
| `tensor_cmp/engines/engine_0/*.pt` | SGLang inference 每层 mlp.output（dumper 产生） |

logprob 文件由 `MILES_TRUE_ON_POLICY_SAVE_DIR` 机制产生；tensor_cmp 文件由 `DUMPER_ENABLE=True` 产生。

**Hidden state 对比**（自动，当 `tensor_cmp/` 存在时）

从 `tensor_cmp/fwd_only/` 和 `tensor_cmp/engines/engine_0/` 读取 `.pt` 文件，按 `layer_id` 分组，按 `step` 排序后 concat，得到 `[total_tokens, hidden_dim]`。对每层计算 MSE 和 mean L2 diff。

注意：Megatron 是 full-sequence batch forward，SGLang 是 prefill(step=0) + 每 decode step 一个 token。两者 concat 后 token 总数应相同，但**顺序可能不同**（序列长度不一致时）。MSE 仅在形状和顺序均一致时有意义。

**Mode 2 — dump_details .pt**

```bash
python tools/true-on-policy/compute_metrics.py \
    --save-dir /root/true-on-policy/20260708_143022 \
    --dump-details /root/output/<run_id>/dump_details
python tools/true-on-policy/compute_metrics.py \
    --save-dir /root/true-on-policy/20260708_143022 \
    --dump-details /root/output/<run_id>/dump_details --rank 0 --rollout 0 1
```

读取 `dump_details/train_data/{rollout_id}_{rank}.pt`，从 `RolloutBatch` 中提取 `log_probs` 和 `rollout_log_probs`（list of tensors → concat）。由 dump_details 开关产生，需 Miles 可 import。

**计算原理**

所有 rollout 按序 concat 后统一计算：

| 指标 | 公式 |
|------|------|
| Pearson r | `dot(a-ā, b-b̄) / sqrt(‖a-ā‖²·‖b-b̄‖²)`，float64；理论值 1.0 |
| MSE | `mean((log_probs - rollout_log_probs)²)` |
| mean/max/p99 \|diff\| | 逐 token 绝对差统计 |
| per-layer mean L2 | 每层 `[total_tokens, hidden_dim]` 按 token 算 L2 norm 后取均值 |
| Cumulative MSE | `mean(per_layer_mean_L2²)` |

**写入文件**

追加到 `<save_dir>/metrics/match.csv`（首次写 header）：

```
timestamp, model_name, batch_size, max_response_len, num_tokens, pearson_r, mse, mean_abs_diff, max_abs_diff, p99_abs_diff
```

`run_megatron.py` 运行结束后自动打印带 `--model-name / --batch-size / --max-response-len` 的完整命令。

Hidden state 对比 CSV `metrics/train_rollout_hidden_states.csv`：

```
timestamp, layer_id, megatron_shape, sglang_shape, mse, mean_l2_diff, cosine_sim, mean_l2_megatron, mean_l2_sglang
```

shape 不匹配时 `mse` / `mean_l2_diff` 为空，`cosine_sim` / `mean_l2_megatron` / `mean_l2_sglang` 仍会计算（基于均值向量和每 token L2 norm，不需要逐 token 对齐）。

## 落盘文件结构与数据来源

### `/root/true-on-policy/{YYYYMMDD_HHMMSS}/`（每次运行一个目录）

```
rollout_0000/
  log_probs.npy           # [total_tokens] float32 — Megatron 重算 logprobs
  rollout_log_probs.npy   # [total_tokens] float32 — SGLang 生成时 logprobs
megatron_hs/
  rank_0/
    layer_000.npy         # [total_tokens, hidden_dim] float32
    layer_001.npy
    ...
metrics/
  match.csv               # compute_metrics.py 输出，每次追加一行
tensor_cmp/
  fwd_only/               # Megatron log-prob pass 每层 hidden states
  engines/engine_0/       # SGLang inference 每层 hidden states
```

| 文件                                 | 生产者                                  | 调用链                                                                                                                  |
| ------------------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `rollout_*/log_probs.npy`          | `log_utils.py:_maybe_save_logprobs()` | `actor.py:train()` → `log_rollout_data()` → `_maybe_save_logprobs()`，由 `MILES_TRUE_ON_POLICY_SAVE_DIR` 触发 |
| `rollout_*/rollout_log_probs.npy`  | 同上                                    | 同上                                                                                                                    |
| `megatron_hs/rank_0/layer_NNN.npy` | `megatron_hs_hook.py:register()`      | `model.py:compute_log_probs()` 前调用，`atexit` 写盘，由 `--custom-megatron-before-log-prob-hook-path` 注入       |

只有 TP rank=0、PP last stage 写文件。

### `/root/output/{run_id}/`（OUTPUT_DIR）

```
checkpoints/              # Megatron 模型权重（match 模式权重不变）
dump_details/
  rollout_data/{rollout_id}.pt   # {rollout_id, samples: [Sample.to_dict()]}
  train_data/{rollout_id}_{rank}.pt  # {rollout_id, rank, rollout_data: RolloutBatch}
```

| 文件                  | 生产者                                          | 调用链                                                                                                              |
| --------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `rollout_data/*.pt` | `debug_data.py:save_debug_rollout_data()`     | `rollout_manager.py:rollout()` 完成后，含每条 Sample 的 `rollout_log_probs`、tokens、rewards 等                 |
| `train_data/*.pt`   | `train_dump_utils.py:save_debug_train_data()` | `actor.py:train()` 末尾，含完整 `RolloutBatch`（同时有 `log_probs` 和 `rollout_log_probs`），可直接对比两者 |

### `tensor_cmp/`（位于运行目录下，由运行时推导）

```
fwd_only/                 # Megatron log-prob pass 每层 hidden states（重算时）
engines/engine_0/         # SGLang inference 每层 hidden states（生成时）
```

| 目录                  | 生产者                                           | 调用链                                                                                                                                                                                 |
| --------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fwd_only/`         | `dumper_utils.py:DumperMegatronUtil(FWD_ONLY)` | `model.py:compute_log_probs()` 中 `DumperMegatronUtil` 注入 PyTorch forward hook，`--dumper-fwd-only enable=true` 触发                                                           |
| `engines/engine_0/` | SGLang 内置`sglang.srt.debug_utils.dumper`     | `server_group.py` 启动时注入 `DUMPER_SERVER_PORT` 环境变量，`sglang_rollout.py:configure_sglang()` 通过 HTTP `/dumper/configure` 激活，`--dumper-inference enable=true` 触发 |

## 环境变量

| 变量                              | 设置方                       | 效果                             |
| --------------------------------- | ---------------------------- | -------------------------------- |
| `MILES_TRUE_ON_POLICY_SAVE_DIR` | `run_megatron.py` 自动设置 | 触发 logprob + hidden state 落盘 |
