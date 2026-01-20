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
MERGED_ROOT="/opt/tiger/hqz_debug/mrx/R1-AL/utils/merged"
SWIFT_CMD="swift"

mkdir -p "${OUTPUT_JSONL}" "${MERGED_ROOT}"

# ================== 3 个 benchmark 配置（各自 JSONL_PREFIX） ==================

# 1) MMAR
INFER_MMAR="python infer_mmar_batch.py"
EVAL_MMAR="python ./eval/mmar_eval_CoT.py"
RESULT_MMAR_TXT="/mnt/hdfs/if_au/saves/mrx/results/result_mmar.txt"
JSONL_PREFIX_MMAR="mmar_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX_MMAR="_1.03.jsonl"

# 2) MMAU NEW
INFER_MMAU_NEW="python infer_mmau_new_batch.py"
EVAL_MMAU_NEW="python ./eval/mmau_eval_CoT.py"
RESULT_MMAU_NEW_TXT="/mnt/hdfs/if_au/saves/mrx/results/result_mmau_new.txt"
JSONL_PREFIX_MMAU_NEW="mmau_new_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX_MMAU_NEW="_1.02.jsonl"

# 3) MMAU OLD
INFER_MMAU_OLD="python infer_mmau_old_batch.py"
EVAL_MMAU_OLD="python ./eval/mmau_eval_CoT.py"
RESULT_MMAU_OLD_TXT="/mnt/hdfs/if_au/saves/mrx/results/result_mmau_old.txt"
JSONL_PREFIX_MMAU_OLD="mmau_old_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX_MMAU_OLD="_1.02.jsonl"

touch "${RESULT_MMAR_TXT}" "${RESULT_MMAU_NEW_TXT}" "${RESULT_MMAU_OLD_TXT}"

LOG_DIR="/mnt/hdfs/if_au/saves/mrx/result/logs_all3"
mkdir -p "${LOG_DIR}"

run_benchmark () {
  local bench_name="$1"
  local infer_cmd="$2"
  local eval_cmd="$3"
  local result_txt="$4"
  local jsonl_file="$5"
  local exp_prefix="$6"
  local step="$7"

  echo "[${bench_name}] 推理..."
  ${infer_cmd} --model_path "${merged_dir}" --output "${jsonl_file}"

  if [[ ! -f "${jsonl_file}" ]]; then
    echo "[${bench_name}][STEP ${step}] ERROR: 未找到推理输出 ${jsonl_file}"
    exit 1
  fi

  echo "[${bench_name}] 评测..."
  local eval_out
  eval_out="$(${eval_cmd} --input "${jsonl_file}")"

  echo "[${bench_name}] 写入 ${result_txt}（追加，带锁）..."
  {
    flock -w 600 200
    while IFS= read -r line; do
      printf "%s %s %s %s\n" "${exp_prefix}" "${step}" "${bench_name}" "${line}"
    done <<< "${eval_out}"
  } 200>>"${result_txt}"
}

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

  local parent_dir version_dir run_tag version_tag exp_prefix
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
    local step lora_dir safe_prefix
    step="${ckpt#checkpoint-}"
    lora_dir="${BASE_DIR}/${ckpt}"

    merged_dir="${MERGED_ROOT}/checkpoint_${run_tag}_${version_tag}_${step}"
    safe_prefix="${exp_prefix//\//_}"

    # ★ 这里用各自 JSONL_PREFIX_*
    jsonl_mmar="${OUTPUT_JSONL}/${safe_prefix}_${JSONL_PREFIX_MMAR}${step}${JSONL_SUFFIX_MMAR}"
    jsonl_mmau_new="${OUTPUT_JSONL}/${safe_prefix}_${JSONL_PREFIX_MMAU_NEW}${step}${JSONL_SUFFIX_MMAU_NEW}"
    jsonl_mmau_old="${OUTPUT_JSONL}/${safe_prefix}_${JSONL_PREFIX_MMAU_OLD}${step}${JSONL_SUFFIX_MMAU_OLD}"

    echo "============================================================"
    echo "[GPU ] ${GPU_ID}"
    echo "[EXP ] ${exp_prefix}"
    echo "[STEP] ${step}"
    echo "[LoRA] ${lora_dir}"
    echo "[MERG] ${merged_dir}"
    echo "[JSON] ${jsonl_mmar}"
    echo "[JSON] ${jsonl_mmau_new}"
    echo "[JSON] ${jsonl_mmau_old}"

    echo "[STEP ${step}] merge LoRA..."
    ${SWIFT_CMD} export \
      --adapters "${lora_dir}" \
      --merge_lora true \
      --output_dir "${merged_dir}"

    if [[ ! -d "${merged_dir}" ]] || [[ -z "$(ls -A "${merged_dir}" 2>/dev/null)" ]]; then
      echo "[STEP ${step}] ERROR: merge 失败或输出为空：${merged_dir}"
      exit 1
    fi

    # 同卡顺序跑 3 个 benchmark
    run_benchmark "MMAR"     "${INFER_MMAR}"     "${EVAL_MMAR}"     "${RESULT_MMAR_TXT}"     "${jsonl_mmar}"     "${exp_prefix}" "${step}"
    run_benchmark "MMAU_NEW" "${INFER_MMAU_NEW}" "${EVAL_MMAU_NEW}" "${RESULT_MMAU_NEW_TXT}" "${jsonl_mmau_new}" "${exp_prefix}" "${step}"
    run_benchmark "MMAU_OLD" "${INFER_MMAU_OLD}" "${EVAL_MMAU_OLD}" "${RESULT_MMAU_OLD_TXT}" "${jsonl_mmau_old}" "${exp_prefix}" "${step}"

    echo "[STEP ${step}] 删除 merged 目录: ${merged_dir}"
    rm -rf "${merged_dir}"

    echo "[STEP ${step}] 完成"
  done
}

# 并行：一张卡一个 BASE_DIR
pids=()
for i in "${!BASE_DIRS[@]}"; do
  dir="${BASE_DIRS[$i]}"
  gpu="${i}"   # 0..4
  log="${LOG_DIR}/gpu${gpu}.log"

  run_one_dir "${dir}" "${gpu}" > "${log}" 2>&1 &
  pids+=("$!")
  echo "Launched GPU ${gpu} for ${dir} (pid=${pids[-1]}), log=${log}"
done

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

echo
echo "全部 BASE_DIR 并行处理完成 ✅"
echo "结果汇总文件："
echo "  - ${RESULT_MMAR_TXT}"
echo "  - ${RESULT_MMAU_NEW_TXT}"
echo "  - ${RESULT_MMAU_OLD_TXT}"
echo "日志目录：${LOG_DIR}"
