#!/usr/bin/env bash
set -euo pipefail

# =============== 只跑这一个目录 ===============
BASE_DIR="/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R1234/v0-20260110-185215"

OUTPUT_JSONL="/mnt/hdfs/if_au/saves/mrx/results/jsonl_mmar"
MERGED_ROOT="/opt/tiger/hqz_debug/mrx/R1-AL/utils/mergeds"
SWIFT_CMD="swift"

mkdir -p "${OUTPUT_JSONL}" "${MERGED_ROOT}"

# ================== 只保留 MMAR 配置 ==================
INFER_MMAR="python infer_mmar_batch.py"
EVAL_MMAR="python ./eval/mmar_eval_CoT.py --input "
RESULT_MMAR_TXT="/mnt/hdfs/if_au/saves/mrx/results/result_mmar.txt"
JSONL_PREFIX_MMAR="mmar_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX_MMAR="_1.03.jsonl"

LOG_DIR="/mnt/hdfs/if_au/saves/mrx/results/logs_1dir_last17_take12_mmar"
mkdir -p "${LOG_DIR}"

run_mmar () {
  local jsonl_file="$1"
  local exp_prefix="$2"
  local step="$3"
  local merged_dir="$4"

  echo "[MMAR] 推理..."
  ${INFER_MMAR} --model_path "${merged_dir}" --output "${jsonl_file}"

  if [[ ! -f "${jsonl_file}" ]]; then
    echo "[MMAR][STEP ${step}] ERROR: 未找到推理输出 ${jsonl_file}"
    exit 1
  fi

  echo "[MMAR] 评测..."
  local eval_out
  eval_out="$(${EVAL_MMAR} --input "${jsonl_file}")"

  echo "[MMAR] 写入 ${RESULT_MMAR_TXT}（追加，带锁）..."
  {
    flock -w 600 200
    while IFS= read -r line; do
      printf "%s %s %s %s\n" "${exp_prefix}" "${step}" "MMAR" "${line}"
    done <<< "${eval_out}"
  } 200>>"${RESULT_MMAR_TXT}"
}

run_one_ckpt () {
  local BASE_DIR="$1"
  local ckpt="$2"     # e.g. checkpoint-12345
  local GPU_ID="$3"

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export OMP_NUM_THREADS=8

  local parent_dir version_dir run_tag version_tag exp_prefix
  parent_dir="$(basename "$(dirname "${BASE_DIR}")")"
  version_dir="$(basename "${BASE_DIR}")"

  run_tag="$(echo "${parent_dir}" | sed -E 's/.*_(R[0-9A-Za-z_]+)/\1/')"
  version_tag="$(echo "${version_dir}" | sed -E 's/^(v[0-9]+).*/\1/')"
  exp_prefix="${run_tag}/${version_tag}"

  local step lora_dir merged_dir safe_prefix
  step="${ckpt#checkpoint-}"
  lora_dir="${BASE_DIR}/${ckpt}"

  merged_dir="${MERGED_ROOT}/checkpoint_${run_tag}_${version_tag}_${step}"
  safe_prefix="${exp_prefix//\//_}"

  local jsonl_mmar
  jsonl_mmar="${OUTPUT_JSONL}/${safe_prefix}_${JSONL_PREFIX_MMAR}${step}${JSONL_SUFFIX_MMAR}"

  echo
  echo "============================================================"
  echo "[GPU ] ${GPU_ID}"
  echo "[EXP ] ${exp_prefix}"
  echo "[CKPT] ${ckpt}"
  echo "[STEP] ${step}"
  echo "[LoRA] ${lora_dir}"
  echo "[MERG] ${merged_dir}"
  echo "[JSON] ${jsonl_mmar}"
  echo "============================================================"

  echo "[STEP ${step}] merge LoRA..."
  ${SWIFT_CMD} export \
    --adapters "${lora_dir}" \
    --merge_lora true \
    --output_dir "${merged_dir}"

  if [[ ! -d "${merged_dir}" ]] || [[ -z "$(ls -A "${merged_dir}" 2>/dev/null)" ]]; then
    echo "[STEP ${step}] ERROR: merge 失败或输出为空：${merged_dir}"
    exit 1
  fi

  # 只跑 MMAR
  run_mmar "${jsonl_mmar}" "${exp_prefix}" "${step}" "${merged_dir}"

  echo "[STEP ${step}] 删除 merged 目录: ${merged_dir}"
  rm -rf "${merged_dir}"

  echo "[STEP ${step}] 完成"
}

# ================== 主流程：选 CKPT（倒数17到倒数6，共12个） ==================
if [[ ! -d "${BASE_DIR}" ]]; then
  echo "ERROR: BASE_DIR not found: ${BASE_DIR}"
  exit 1
fi

mapfile -t ALL_CKPTS < <(
  find "${BASE_DIR}" -maxdepth 1 -type d -name "checkpoint-[0-9]*" -printf "%f\n" \
  | sed -E 's/^checkpoint-([0-9]+)$/\1 checkpoint-\1/' \
  | sort -n -k1,1 \
  | awk '{print $2}'
)

n="${#ALL_CKPTS[@]}"
if (( n < 17 )); then
  echo "ERROR: checkpoint 数量不足 17 个（当前 ${n} 个），无法取倒数第17~倒数第6。"
  exit 1
fi

# 取倒数 17 个里，去掉最新 5 个 => 共 12 个（倒数第17~倒数第6）
mapfile -t CKPTS < <(printf "%s\n" "${ALL_CKPTS[@]}" | tail -n 17 | head -n 12)

echo "将要测试的 12 个 checkpoints（倒数第17~倒数第6）："
printf "  %s\n" "${CKPTS[@]}"

# ================== 4 卡分 batch：每个 batch 4 个 ckpt ==================
fail=0
batch_id=0

for ((offset=0; offset<${#CKPTS[@]}; offset+=4)); do
  batch_id=$((batch_id+1))
  echo
  echo "##########################"
  echo "### BATCH ${batch_id}"
  echo "##########################"

  pids=()

  for gpu in 0 1 2 3; do
    idx=$((offset+gpu))
    if (( idx >= ${#CKPTS[@]} )); then
      continue
    fi

    ckpt="${CKPTS[$idx]}"
    log="${LOG_DIR}/batch${batch_id}_gpu${gpu}_${ckpt}.log"

    run_one_ckpt "${BASE_DIR}" "${ckpt}" "${gpu}" > "${log}" 2>&1 &
    pids+=("$!")
    echo "Launched batch${batch_id} GPU ${gpu} for ${ckpt} (pid=${pids[-1]}), log=${log}"
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done

  if [[ "${fail}" -ne 0 ]]; then
    echo "BATCH ${batch_id} 有任务失败了 ❌，请查看 ${LOG_DIR}/batch${batch_id}_gpu*.log"
    exit 1
  fi

  echo "BATCH ${batch_id} 完成 ✅"
done

echo
echo "全部 12 个 checkpoint MMAR 测试完成 ✅"
echo "结果汇总文件："
echo "  - ${RESULT_MMAR_TXT}"
echo "日志目录：${LOG_DIR}"
