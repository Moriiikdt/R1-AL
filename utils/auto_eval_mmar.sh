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
INFER_PY="python infer_mmar_batch.py"
EVAL_PY="./eval/mmar_eval_CoT.py"

# 评测结果统一追加到这个文件（不会覆盖）
RESULT_TXT="./result_mmar.txt"

# swift 命令（如果 swift 不在 PATH，请写绝对路径）
SWIFT_CMD="swift"

# infer 脚本生成的 jsonl 文件命名规则
JSONL_PREFIX="mmar_step_review_4K6-3e-sft-RL-"
JSONL_SUFFIX="_1.03.jsonl"

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
  # 取 BASE_DIR 的倒数第 2、1 级目录名
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
  # 找出 checkpoint-数字 形式的目录
  # 按数字大小排序，保证顺序：2800 -> 3000 -> 3200 ...
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
    # 提取 step 数字，如 checkpoint-2800 -> 2800
    step="${ckpt#checkpoint-}"

    # LoRA 目录
    lora_dir="${BASE_DIR}/${ckpt}"

    # merge 后模型目录（swift export 自动生成）
    merged_dir="${BASE_DIR}/${ckpt}-merged"

    # 推理输出的 jsonl 文件名
    jsonl_file="${BASE_DIR}/${JSONL_PREFIX}${step}${JSONL_SUFFIX}"

    echo "============================================================"
    echo "[EXP ] ${exp_prefix}"
    echo "[STEP] ${step}"
    echo "[LoRA] ${lora_dir}"
    echo "[MERG] ${merged_dir}"
    echo "[JSON] ${jsonl_file}"

    # ---------- 1) 合并 LoRA ----------
    # swift export 会在 LoRA 同级目录生成 checkpoint-XXXX-merged
    if [[ -d "${merged_dir}" ]]; then
      echo "[STEP ${step}] 已存在 merged 目录，跳过 merge"
    else
      echo "[STEP ${step}] 开始 merge LoRA..."
      (
        # 切到 BASE_DIR，确保 merged 输出在同级目录
        cd "${BASE_DIR}"
        ${SWIFT_CMD} export \
          --adapters "${lora_dir}" \
          --merge_lora true
      )
    fi

    # merge 后目录必须存在
    if [[ ! -d "${merged_dir}" ]]; then
      echo "[STEP ${step}] ERROR: merge 失败，未找到 ${merged_dir}"
      exit 1
    fi

    # ---------- 2) 推理 ----------
    # 使用合并后的模型做推理，生成 jsonl
    echo "[STEP ${step}] 开始推理..."
    ${INFER_PY} --model_path "${merged_dir}"

    # 检查推理输出是否存在
    if [[ ! -f "${jsonl_file}" ]]; then
      echo "[STEP ${step}] ERROR: 未找到推理输出 ${jsonl_file}"
      exit 1
    fi

    # ---------- 3) 评测 ----------
    # 评测脚本默认输出到 stdout，这里用变量接住
    echo "[STEP ${step}] 开始评测..."
    eval_out="$(${EVAL_PY} --input "${jsonl_file}")"

    # ---------- 4) 追加写入结果 ----------
    # 每一行结果前面加上 exp_prefix + step，方便之后对齐不同实验/版本
    echo "[STEP ${step}] 写入 ${RESULT_TXT}（追加，不覆盖）..."
    while IFS= read -r line; do
      printf "%s %s %s\n" "${exp_prefix}" "${step}" "${line}" >> "${RESULT_TXT}"
    done <<< "${eval_out}"

    # ---------- 5) 清理 merged 目录 ----------
    # 节省磁盘空间，每测完一个就删掉合并后的模型
    echo "[STEP ${step}] 删除 merged 目录: ${merged_dir}"
    rm -rf "${merged_dir}"

    echo "[STEP ${step}] 完成"
  done
done

echo
echo "全部 BASE_DIR 处理完成 ✅"
echo "结果汇总文件：${RESULT_TXT}"
