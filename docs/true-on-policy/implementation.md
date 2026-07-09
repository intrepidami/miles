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

注：`--true-on-policy-mode` **不再**是 match 模式的默认参数。当前 `run_megatron.py` 设置 `true_on_policy=False`，**不**附加 `--true-on-policy-mode`，跑的是基线失配测量。`--true-on-policy-mode` 本身不设置 `--use-rollout-logprobs`，所以 `actor.py` 中 `not args.use_rollout_logprobs` 已为 True，Megatron 无条件重算 logprobs，无需 `--get-mismatch-metrics`。

`true_on_policy=True` 时 `build_true_on_policy_launch_plan`（`miles/true_on_policy/config.py`）自动附加 `--recompute-logprobs-via-prefill`（rollout 结束后 SGLang 对完整序列做一次 prefill 重算，覆盖 decode 时的 logprobs）、SGLang 确定性推理参数、Megatron 确定性 kernel 参数及相关 env vars。

## true-on-policy 在 Qwen3-0.6B 单卡 match 下的生效面

`true_on_policy=True` 时 `build_true_on_policy_launch_plan`（`miles/true_on_policy/config.py`）对 Qwen3-0.6B（profile `qwen3_dense`，contract `qwen3_dense_true_on_policy_v1`）+ Megatron + 单卡（TP=PP=CP=1，rollout 单卡引擎 → `sglang_target="fsdp"`）生成的开关及其实际状态：

| 开关 | 来源 | 单卡是否生效 |
| --- | --- | --- |
| `--recompute-logprobs-via-prefill`：rollout 后 SGLang prefill 重算 `rollout_log_probs`（`prefill_logprobs.py`），消除 decode/prefill 路径差异 | `miles_args` | 生效 |
| `--true-on-policy-mode`：训练侧 logits 转 bf16 再算 logprob（`logit_processors.py:56`）；`rollout_log_probs` 以 bf16 存储（`data.py:22`），两侧数值精度对齐 | `miles_args` | 生效 |
| `--deterministic-mode`（Megatron 确定性执行） | `miles_args` | 生效 |
| `--sglang-enable-deterministic-inference` + `--sglang-attention-backend fa3` + `--sglang-true-on-policy-contract` | `sglang_args` | 生效 |
| `--transformer-impl local` + `--true-on-policy-contract`：Megatron 层 spec 换用 SGLang math kernel（`model_provider.py:250` `use_true_on_policy_backend`）；`--batch-invariant-mode`、`--no-rope-fusion`、`--no-bias-swiglu-fusion` | `megatron_args` | 生效 |
| `tp_invariant_row_linear` / `deterministic_tp_allreduce`（跨 rank TP 数值一致性） | kernel policy | **不生效** — 仅 `sglang_target="fsdp_tp"`（TP>1）时开启 |
| `use_sequence_parallel=False`（script defaults） | `apply_true_on_policy_script_defaults` | 无效果 — SP 本需 TP>1 |
| `NCCL_ALGO=Ring`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`、`CUBLAS_WORKSPACE_CONFIG` | env vars | `NCCL_ALGO` 单卡无通信、无效果；其余生效 |

即：单卡下 true-on-policy 的全部有效改动 = **kernel 对齐（Megatron 用 SGLang math kernel + 关 fusion + batch invariant）+ prefill 重算 + bf16 精度对齐 + 双侧确定性**。TP 相关机制全部闲置。

### 为什么 Pearson 对开关不敏感

**设定**。记 `a` = Megatron 重算的 logprobs，`b` = SGLang 的 rollout logprobs，逐 token 配对。把失配建模为 `b = a + ε`：ε 是两引擎的数值差异（kernel 归约顺序、精度、fusion 等造成），近似与 a 独立、均值≈0，因此 `MSE ≈ Var(ε)`。

**用到的事实**。① `cov(x,x) = Var(x)`；② ε 与 a 独立时 `cov(a,ε) = 0` 且 `Var(a+ε) = Var(a) + Var(ε)`；③ Pearson 定义 `r = cov(a,b)/(σ_a·σ_b)`，其中 `σ = √Var`；④ ε 均值≈0 时 `MSE = mean(ε²) ≈ Var(ε)`。

**推导（三步）**：

```
第1步 分子：  cov(a,b) = cov(a, a+ε) = cov(a,a) + cov(a,ε) = Var(a) + 0 = Var(a)
第2步 分母：  Var(b) = Var(a+ε) = Var(a) + Var(ε)，故 σ_b = √(Var(a)+Var(ε))
第3步 代入：  r = Var(a) / [√Var(a) · √(Var(a)+Var(ε))]
                = 1 / √(1 + x)，   x ≜ Var(ε)/Var(a)（噪声占信号比例，精确式）
```

x ≪ 1 时取一阶近似 `1/√(1+x) ≈ 1 − x/2`（x<0.01 时误差可忽略），再用④把 `Var(ε)` 换成 MSE：

```
r ≈ 1 − MSE / (2·Var(a))
```

**第 3 步细节：√/√ 的合并**。规则 `√a/√b = √(a/b)`（a≥0、b>0：设 y=√a/√b，则 y²=a/b 且 y≥0，即 y=√(a/b)）。完整过程——先用 `Var(a) = √Var(a)·√Var(a)` 约掉分母中一个因子：

```
      Var(a)                     √Var(a)
─────────────────────── = ───────────────── = √(Var(a)/(Var(a)+Var(ε))) = √(1/(1+x)) = 1/√(1+x)
√Var(a)·√(Var(a)+Var(ε))   √(Var(a)+Var(ε))
```

**近似的来源：为什么 `1/√(1+x) ≈ 1 − x/2`**。两种看法：

- 切线近似：`f(x) = (1+x)^(-1/2)`，`f'(0) = −½`，在 x=0 处用切线代替曲线得 `f(x) ≈ 1 − x/2`。一般规则 `(1+x)^n ≈ 1 + n·x`，此处 n = −½——公式里"除以 **2**·Var(a)"的 2 即由此来。
- 平方验证（不用微积分）：两边平方后比较，`1/(1+x) ≈ 1−x`（因 `(1−x)(1+x)=1−x²`）与 `(1−x/2)² = 1−x+x²/4 ≈ 1−x` 一致，误差均为 x² 量级。

数值验证：x=0.01 时精确值 `1/√1.01 = 0.9950372`，近似值 `0.9950000`，误差 3.7e-5 ≈ x²。本场景 x = MSE/Var(a) ~ 1e-5 以下，误差 ~1e-10，近似即精确。

**信号与噪声的量级差**。logprob 本身分布很宽：高置信 token 接近 0，难 token 掉到 −5、−10 以下。这个跨度是"信号"——两个引擎对同一批 token 给出的共同宽分布，样本方差 `Var(a)` 量级 1–10 nat²。失配 ε 只是叠加其上的微扰：基线（不开 true-on-policy）逐 token 差通常 ~1e-3–1e-2，MSE ~1e-6–1e-4，比 `Var(a)` 小 4–7 个数量级。

**代入数字**（取 `Var(a) = 2`）：

| 配置 | MSE | r = 1 − MSE/(2·Var) | 打印 6 位小数 |
| --- | --- | --- | --- |
| 基线（关） | 1e-4 | 0.999975 | 0.999975 |
| true-on-policy（开） | 1e-8 | 0.9999999975 | 1.000000 |

MSE 降 4 个数量级，r 只从小数点后第 5 位挪到第 9 位——开关前后 r 几乎不可辨。

**直觉**。Pearson 本质是信噪比度量：`1 − r ∝ 噪声方差/信号方差`。信号（token 难易差异）巨大，噪声被它一除就湮灭。好比两把尺子量 0–1m 的物体，误差 0.01mm 还是 0.1mm，相关系数都 ≈1——相关性对误差的绝对大小天然钝感。MSE / mean|diff| 直接度量 ε 本身，不被信号方差归一化，所以能区分开关。

结论：Pearson 只适合做结构性 sanity check（r 明显 <0.99 说明 token 错位等结构问题——错位使 ε 不再是小扰动而是打乱配对，`cov(a,b)` 崩塌）；**判别 true-on-policy 效果要看 MSE / mean|diff| / max|diff|**，它们随开关变化可差数个数量级。`match.csv` 已包含这些字段。

## 在线 Metric：train_rollout_logprob_abs_diff

`losses.py:policy_loss_function` 在每次训练 forward 中计算（`rollout_log_probs` 存在时）：

| 变量                       | 来源                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `train_scored_log_probs` | `batch["log_probs"]`（Megatron 完整序列重算）                                                     |
| `rollout_log_probs`      | `batch["rollout_log_probs"]`（SGLang prefill 重算，由 `--recompute-logprobs-via-prefill` 触发） |

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

模式自动推导：`{save_dir}/dump_details/train_data/` 存在时用 Mode 2（dump_details），否则用 Mode 1（.npy）。CSV 始终写 `<save_dir>/metrics/match.csv`。

**参数**

| 参数                 | 类型           | 说明                                                |
| -------------------- | -------------- | --------------------------------------------------- |
| `--save-dir PATH`  | 必填           | 运行目录（`BASE_DIR/YYYYMMDD_HHMMSS/`）；始终写 CSV |
| `--rank INT`       | 可选，默认 0   | dump_details 模式：读 `{rollout_id}_{rank}.pt`    |
| `--rollout INT...` | 可选，默认全部 | 指定 rollout ID，如 `--rollout 0 1`               |
| `--json`           | flag           | 额外打印 JSON                                       |
| `--plot`           | flag           | 生成 `metrics/logprob_scatter.png`                |

```bash
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022
python tools/true-on-policy/compute_metrics.py --save-dir /root/true-on-policy/20260708_143022 --rollout 0
```

读取文件：

| 文件                                 | 作用                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `rollout_*/log_probs.npy`          | Megatron 重算 logprobs，`[total_tokens]` float32    |
| `rollout_*/rollout_log_probs.npy`  | SGLang 生成时 logprobs，`[total_tokens]` float32    |
| `tensor_cmp/fwd_only/*.pt`         | Megatron log-prob pass 每层 mlp.output（dumper 产生） |
| `tensor_cmp/engines/engine_0/*.pt` | SGLang inference 每层 mlp.output（dumper 产生）       |

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

**加载时的对齐校验**（自动执行）：

- Mode 1：每个 rollout 目录内 `log_probs.npy` 与 `rollout_log_probs.npy` shape 必须相等（逐 rollout 检查，不只靠 concat 后的全局 assert——两个 rollout 的错位可能互相抵消）。
- Mode 2：逐 sample 检查 `log_probs[i].shape == rollout_log_probs[i].shape`；当 token 总数等于 `sum(response_lengths)` 时，进一步检查每个 sample 的 token 数等于 `response_lengths[i]`（总数不等视为 CP 分片，跳过该项并打印提示；缺 `response_lengths` 同样跳过并提示）。全局 size 相等但逐 sample 错位（一长一短互补）会被此检查捕获。

**计算原理**

所有 rollout 按序 concat 后统一计算：

| 指标                 | 公式                                                                      |
| -------------------- | ------------------------------------------------------------------------- |
| Pearson r            | `dot(a-ā, b-b̄) / sqrt(‖a-ā‖²·‖b-b̄‖²)`，float64；理论值 1.0 |
| MSE                  | `mean((log_probs - rollout_log_probs)²)`                               |
| mean/max/p99\|diff\| | 逐 token 绝对差统计                                                       |
| per-layer mean L2    | 每层`[total_tokens, hidden_dim]` 按 token 算 L2 norm 后取均值           |
| Cumulative MSE       | `mean(per_layer_mean_L2²)`                                             |

**写入文件**

写入 `<save_dir>/metrics/match.csv`（每次覆盖）：

```
timestamp, num_tokens, pearson_r, mse, mean_abs_diff, max_abs_diff, p99_abs_diff
```

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
  match.csv               # compute_metrics.py 输出，每次覆盖
tensor_cmp/
  fwd_only/               # Megatron log-prob pass 每层 hidden states
  engines/engine_0/       # SGLang inference 每层 hidden states
```

| 文件                                 | 生产者                                  | 调用链                                                                                                                  |
| ------------------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `rollout_*/log_probs.npy`          | `log_utils.py:_maybe_save_logprobs()` | `actor.py:train()` → `log_rollout_data()` → `_maybe_save_logprobs()`，由 `MILES_TRUE_ON_POLICY_SAVE_DIR` 触发 |
| `rollout_*/rollout_log_probs.npy`  | 同上                                    | 同上                                                                                                                    |
| `megatron_hs/rank_0/layer_NNN.npy` | `megatron_hs_hook.py:register()`      | `model.py:compute_log_probs()` 前调用，`atexit` 写盘，由 `--custom-megatron-before-log-prob-hook-path` 注入       |

只有 TP rank=0、PP last stage、intra_dp_cp rank=0 写文件（避免 DP/CP>1 时多 rank 写同一文件竞态）。

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
