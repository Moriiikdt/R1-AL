#!/usr/bin/env bash
# 遇到错误立即退出；未定义变量报错；管道中任一命令失败即失败
set -euo pipefail

# ================== 多个 BASE_DIR（按指定顺序） ==================
BASE_DIRS=(
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R1234/v0-20260110-185215"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R123/v1-20260112-021420"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R124/v0-20260112-161256"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R12/v0-20260111-111605"
  "/mnt/hdfs/if_au/saves/mrx/checkpoints/output_step_reason_4K6_R12_simple/v7-20260115-024725"
)

# ================== 通用配置 ==================
# 推理脚本 & 评测脚本（按你的实际路径修改）
INFER_PY="python infer_mmau_old_batch.py"
EVAL_PY="./eval/mmau_eval_CoT.py"

# 评测结果统一追加到这个文件（不会覆盖）
RESULT_TXT="./result_mmau_old.txt"

# swift 命令（如果 swift 不在 PATH，请写绝对路径）
SWIFT_CMD="swift"

# infer 脚本生成的 jsonl 文件命名规则
JSONL_PREFIX="mmau_old_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX="_1.02.jsonl"

echo "RESULT_TXT: ${RESULT_TXT}"
echo "Start..."

# ================== 外层：逐个 BASE_DIR ==================
for BASE_DIR in "${BASE_DIRS[@]}"; do
  echo
  echo "############################################################"
  echo "### Processing BASE_DIR:"
  echo "### ${BASE_DIR}"
  echo "############################################################"

  # BASE_DIR 不存在则跳过
  if [[ ! -d "${BASE_DIR}" ]]; then
    echo "WARNING: BASE_DIR not found, skip: ${BASE_DIR}"
    continue
  fi

  # ---------- 解析实验标识，如 R1234/v0 ----------
  parent_dir="$(basename "$(dirname "${BASE_DIR}")")"   # output_step_reason_4K6_R1234
  version_dir="$(basename "${BASE_DIR}")"                # v0-20260110-185215

  # 提取 R1234 / R12_simple 等
  run_tag="$(echo "${parent_dir}" | sed -E 's/.*_(R[0-9A-Za-z_]+)/\1/')"

  # 提取 v0 / v7 等
  version_tag="$(echo "${version_dir}" | sed -E 's/^(v[0-9]+).*/\1/')"

  # 最终前缀
  exp_prefix="${run_tag}/${version_tag}"
  echo "Experiment tag: ${exp_prefix}"

  # ================== 收集并排序 checkpoint ==================
  mapfile -t CKPTS < <(
    find "${BASE_DIR}" -maxdepth 1 -type d -name "checkpoint-[0-9]*" -printf "%f\n" \
    | sed -E 's/^checkpoint-([0-9]+)$/\1 checkpoint-\1/' \
    | sort -n -k1,1 \
    | awk '{print $2}'
  )

  # 如果当前 BASE_DIR 下没有 checkpoint，就跳过
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "No checkpoint-* found in ${BASE_DIR}, skip."
    continue
  fi

  # ================== 内层：逐个 checkpoint 处理 ==================
  for ckpt in "${CKPTS[@]}"; do
    step="${ckpt#checkpoint-}"

    lora_dir="${BASE_DIR}/${ckpt}"
    merged_dir="${BASE_DIR}/${ckpt}-merged"
    jsonl_file="${BASE_DIR}/${JSONL_PREFIX}${step}${JSONL_SUFFIX}"

    echo "============================================================"
    echo "[EXP ] ${exp_prefix}"
    echo "[STEP] ${step}"
    echo "[LoRA] ${lora_dir}"
    echo "[MERG] ${merged_dir}"
    echo "[JSON] ${jsonl_file}"

    # ---------- 1) 合并 LoRA ----------
    if [[ -d "${merged_dir}" ]]; then
      echo "[STEP ${step}] 已存在 merged 目录，跳过 merge"
    else
      echo "[STEP ${step}] 开始 merge LoRA..."
      (
        cd "${BASE_DIR}"
        ${SWIFT_CMD} export \
          --adapters "${lora_dir}" \
          --merge_lora true
      )
    fi

    if [[ ! -d "${merged_dir}" ]]; then
      echo "[STEP ${step}] ERROR: merge 失败，未找到 ${merged_dir}"
      exit 1
    fi

    # ---------- 2) 推理 ----------
    echo "[STEP ${step}] 开始推理..."
    ${INFER_PY} --model_path "${merged_dir}"

    if [[ ! -f "${jsonl_file}" ]]; then
      echo "[STEP ${step}] ERROR: 未找到推理输出 ${jsonl_file}"
      exit 1
    fi

    # ---------- 3) 评测 ----------
    echo "[STEP ${step}] 开始评测..."
    eval_out="$(${EVAL_PY} --input "${jsonl_file}")"

    # ---------- 4) 追加写入结果 ----------
    # 输出格式：R1234/v0 2800 acc=...
    echo "[STEP ${step}] 写入 ${RESULT_TXT}（追加，不覆盖）..."
    while IFS= read -r line; do
      printf "%s %s %s\n" "${exp_prefix}" "${step}" "${line}" >> "${RESULT_TXT}"
    done <<< "${eval_out}"

    # ---------- 5) 清理 merged 目录 ----------
    echo "[STEP ${step}] 删除 merged 目录: ${merged_dir}"
    rm -rf "${merged_dir}"

    echo "[STEP ${step}] 完成"
  done
done

echo
echo "全部 BASE_DIR 处理完成 ✅"
echo "结果汇总文件：${RESULT_TXT}"
