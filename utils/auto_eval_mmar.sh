#!/usr/bin/env bash
set -euo pipefail

BASE_DIRS=(
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R1234/v0-20260110-185215"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R123/v1-20260112-021420"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R124/v0-20260112-161256"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R12/v0-20260111-111605"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R12_simple/v7-20260115-024725"
)

OUTPUT_JSONL="/mnt/hdfs/if_au/saves/mrx/result"
INFER_PY="python infer_mmar_batch.py"
EVAL_PY="python ./eval/mmar_eval_CoT.py"
RESULT_TXT="/mnt/hdfs/if_au/saves/mrx/result/result_mmar.txt"
SWIFT_CMD="swift"
JSONL_PREFIX="mmar_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX="_1.03.jsonl"

LOG_DIR="/mnt/hdfs/if_au/saves/mrx/result/logs"
mkdir -p "${LOG_DIR}"

# ====== 每个 BASE_DIR 的工作函数（完全复用你原来的逻辑）======
run_one_dir () {
  local BASE_DIR="$1"
  local GPU_ID="$2"

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export OMP_NUM_THREADS=8

  echo
  echo "############################################################"
  echo "### GPU ${GPU_ID} Processing BASE_DIR:"
  echo "### ${BASE_DIR}"
  echo "############################################################"

  if [[ ! -d "${BASE_DIR}" ]]; then
    echo "WARNING: BASE_DIR not found, skip: ${BASE_DIR}"
    return 0
  fi

  parent_dir="$(basename "$(dirname "${BASE_DIR}")")"
  version_dir="$(basename "${BASE_DIR}")"
  run_tag="$(echo "${parent_dir}" | sed -E 's/.*_(R[0-9A-Za-z_]+)/\1/')"
  version_tag="$(echo "${version_dir}" | sed -E 's/^(v[0-9]+).*/\1/')"
  exp_prefix="${run_tag}/${version_tag}"
  echo "Experiment tag: ${exp_prefix}"

  mapfile -t CKPTS < <(
    find "${BASE_DIR}" -maxdepth 1 -type d -name "checkpoint-[0-9]*" -printf "%f\n" \
    | sed -E 's/^checkpoint-([0-9]+)$/\1 checkpoint-\1/' \
    | sort -n -k1,1 \
    | awk '{print $2}' \
    | tail -n 5
  )

  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "No checkpoint-* found in ${BASE_DIR}, skip."
    return 0
  fi

  for ckpt in "${CKPTS[@]}"; do
    step="${ckpt#checkpoint-}"
    lora_dir="${BASE_DIR}/${ckpt}"
    merged_dir="${BASE_DIR}/${ckpt}-merged"
    safe_prefix="${exp_prefix//\//_}"
    jsonl_file="${OUTPUT_JSONL}/${safe_prefix}_${JSONL_PREFIX}${step}${JSONL_SUFFIX}"

    echo "============================================================"
    echo "[GPU ] ${GPU_ID}"
    echo "[EXP ] ${exp_prefix}"
    echo "[STEP] ${step}"
    echo "[LoRA] ${lora_dir}"
    echo "[MERG] ${merged_dir}"
    echo "[JSON] ${jsonl_file}"

    if [[ -d "${merged_dir}" ]]; then
      echo "[STEP ${step}] merged 已存在，跳过 merge"
    else
      echo "[STEP ${step}] 开始 merge LoRA..."
      (
        cd "${BASE_DIR}"
        ${SWIFT_CMD} export --adapters "${lora_dir}" --merge_lora true
      )
    fi

    if [[ ! -d "${merged_dir}" ]]; then
      echo "[STEP ${step}] ERROR: merge 失败，未找到 ${merged_dir}"
      exit 1
    fi

    echo "[STEP ${step}] 开始推理..."
    ${INFER_PY} --model_path "${merged_dir}" --output "${jsonl_file}"

    if [[ ! -f "${jsonl_file}" ]]; then
      echo "[STEP ${step}] ERROR: 未找到推理输出 ${jsonl_file}"
      exit 1
    fi

    echo "[STEP ${step}] 开始评测..."
    eval_out="$(${EVAL_PY} --input "${jsonl_file}")"

    # 关键点：并发追加写 RESULT_TXT 时加锁，避免多进程互相打乱行
    echo "[STEP ${step}] 写入 ${RESULT_TXT}（追加，带锁）..."
    {
      flock -w 600 200
      while IFS= read -r line; do
        printf "%s %s %s\n" "${exp_prefix}" "${step}" "${line}"
      done <<< "${eval_out}"
    } 200>>"${RESULT_TXT}"

    echo "[STEP ${step}] 删除 merged 目录: ${merged_dir}"
    rm -rf "${merged_dir}"

    echo "[STEP ${step}] 完成"
  done
}

echo "RESULT_TXT: ${RESULT_TXT}"
echo "Start..."

# ====== 并行：一张卡一个目录 ======
pids=()
for i in "${!BASE_DIRS[@]}"; do
  dir="${BASE_DIRS[$i]}"
  gpu="${i}"   # 0..4
  log="${LOG_DIR}/gpu${gpu}.log"
  # 每个目录一个后台进程，并记录日志
  run_one_dir "${dir}" "${gpu}" > "${log}" 2>&1 &
  pids+=("$!")
  echo "Launched GPU ${gpu} for ${dir} (pid=${pids[-1]}), log=${log}"
done

# 等待全部结束
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "有任务失败了 ❌，请查看 ${LOG_DIR}/gpu*.log"
  exit 1
fi

echo "全部 BASE_DIR 并行处理完成 ✅"
echo "结果汇总文件：${RESULT_TXT}"
echo "日志目录：${LOG_DIR}"
