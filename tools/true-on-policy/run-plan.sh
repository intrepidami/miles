#!/usr/bin/env bash
# True-on-policy match 批量运行计划：fsdp / megatron × baseline / true-on-policy × 并行度。
# 顺序执行；每个 case 跑完立即 compute_metrics；失败的 case 记 FAILED 不中断后续。
# 结果汇总追加到 $BASE_DIR/run-plan-summary-<timestamp>.txt。
#
# 用法:
#   bash tools/true-on-policy/run-plan.sh 2>&1 | tee run-plan.log
# 按需注释/放开下方 run_case 行。

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR=/root/true-on-policy   # 与 run_match.py 的 BASE_DIR 保持一致
SUMMARY="$BASE_DIR/run-plan-summary-$(date +%Y%m%d_%H%M%S).txt"
cd "$REPO_ROOT"
mkdir -p "$BASE_DIR"

run_case() {
    local name="$1"; shift
    echo ""
    echo "=== [$name] run_match.py $* ==="
    if ! python tools/true-on-policy/run_match.py --num-nodes 1 --skip-prepare "$@"; then
        echo "[$name] FAILED" | tee -a "$SUMMARY"
        return 0
    fi
    # run_match.py 把每次运行存进 BASE_DIR/{YYYYMMDD_HHMMSS}/；顺序执行下最新目录即本次
    local save_dir
    save_dir=$(ls -td "$BASE_DIR"/2*/ 2>/dev/null | head -1)
    python tools/true-on-policy/compute_metrics.py --save-dir "$save_dir"
    {
        echo "[$name] $save_dir"
        [ -f "$save_dir/metrics/match.csv" ] && sed 's/^/    /' "$save_dir/metrics/match.csv"
    } >> "$SUMMARY"
}

# ---------------- FSDP ----------------
run_case fsdp-1gpu-base --cuda-visible-devices 2 --num-gpus-per-node 1 --train-backend fsdp
run_case fsdp-1gpu-top  --cuda-visible-devices 2 --num-gpus-per-node 1 --train-backend fsdp --true-on-policy
run_case fsdp-dp2-base  --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend fsdp
run_case fsdp-dp2-top   --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend fsdp --true-on-policy

# ---------------- Megatron ----------------
run_case mega-1gpu-base --cuda-visible-devices 2 --num-gpus-per-node 1 --train-backend megatron
run_case mega-1gpu-top  --cuda-visible-devices 2 --num-gpus-per-node 1 --train-backend megatron --true-on-policy
# DP=2：forward 数值路径与单卡一致，可直接与 1gpu / fsdp-dp2 三方对比
run_case mega-dp2-base  --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend megatron --megatron-dp 2
run_case mega-dp2-top   --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend megatron --megatron-dp 2 --true-on-policy
# TP/CP 改 kernel 数值路径（TP>1 自动开 sequence-parallel，CP 走 a2a），
# 与单卡/SGLang(TP=1) 口径不可直接比——只测管线正确性（见 README 可选参数）
run_case mega-cp2-base   --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend megatron --megatron-cp 2
run_case mega-cp2-top   --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend megatron --megatron-cp 2 --true-on-policy

run_case mega-tp2-base   --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend megatron --megatron-tp 2
run_case mega-tp2-top   --cuda-visible-devices 6,7 --num-gpus-per-node 2 --train-backend megatron --megatron-tp 2 --true-on-policy
# 8 卡全并行（占满整机，按需放开）
# run_case mega-tp2cp2dp2-top --cuda-visible-devices 0,1,2,3,4,5,6,7 --num-gpus-per-node 8 --train-backend megatron --megatron-tp 2 --megatron-cp 2 --megatron-dp 2 --true-on-policy

echo ""
echo "=== SUMMARY ($SUMMARY) ==="
cat "$SUMMARY"
