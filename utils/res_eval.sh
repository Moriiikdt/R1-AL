# 保存为 run_all_eval.sh
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hdfs/if_au/saves/mrx/results/jsonl_mmar"
LOG_DIR="${ROOT}/_eval_logs_mmar"
mkdir -p "${LOG_DIR}"

MMAR_SCRIPT="./eval/mmar_eval_CoT.py"
MMAU_SCRIPT="./eval/mmau_eval_CoT.py"

# 颜色（可选）
RED=$'\033[0;31m'
GRN=$'\033[0;32m'
YLW=$'\033[0;33m'
NC=$'\033[0m'

# 提取标识：R12_simple / R1234 / R124 / R123 / R12
get_tag() {
  local base="$1"
  if [[ "$base" == R12_simple_* ]]; then
    echo "R12_simple"
  elif [[ "$base" == R1234_* ]]; then
    echo "R1234"
  elif [[ "$base" == R124_* ]]; then
    echo "R124"
  elif [[ "$base" == R123_* ]]; then
    echo "R123"
  elif [[ "$base" == R12_* ]]; then
    echo "R12"
  else
    echo "UNKNOWN"
  fi
}

# 选择 eval 脚本
pick_eval() {
  local base="$1"
  if [[ "$base" == *"_mmar_"* ]]; then
    echo "${MMAR_SCRIPT}"
  elif [[ "$base" == *"_mmau_new_"* ]] || [[ "$base" == *"_mmau_old_"* ]] || [[ "$base" == *"_mmau_"* ]]; then
    # 你说 mmau_new / mmau_old -> mmau_eval_CoT.py
    echo "${MMAU_SCRIPT}"
  else
    echo ""
  fi
}

# 遍历所有 jsonl（按名字排序，保证输出稳定）
mapfile -t FILES < <(find "${ROOT}" -maxdepth 1 -type f -name "*.jsonl" | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "${RED}[ERROR] No .jsonl files found in ${ROOT}${NC}"
  exit 1
fi

echo "${GRN}[INFO] Found ${#FILES[@]} files under ${ROOT}${NC}"
echo "${GRN}[INFO] Logs will be saved to ${LOG_DIR}${NC}"
echo

FAIL=0
SKIP=0
OK=0

for f in "${FILES[@]}"; do
  base="$(basename "$f")"
  tag="$(get_tag "$base")"
  eval_script="$(pick_eval "$base")"

  if [[ -z "${eval_script}" ]]; then
    echo "${YLW}[SKIP] file=${base} tag=${tag} reason=unrecognized(mm type)${NC}"
    ((SKIP+=1))
    continue
  fi

  # log 文件名：加时间戳避免覆盖
  ts="$(date +"%Y%m%d_%H%M%S")"
  log_file="${LOG_DIR}/${base}.${ts}.log"

  # 你要的“结果这样的形式”：这里每个文件都会先打印文件名/标识/用哪个脚本
  echo "============================================================"
  echo "[RUN ] file=${base} tag=${tag} eval=$(basename "${eval_script}")"
  echo "[CMD ] python ${eval_script} --input ${f}"
  echo "------------------------------------------------------------"

  # 执行并把 stdout/stderr 都 tee 到日志
  if python "${eval_script}" --input "${f}" 2>&1 | tee "${log_file}"; then
    echo "[DONE] file=${base} tag=${tag} status=OK log=${log_file}"
    ((OK+=1))
  else
    echo "${RED}[DONE] file=${base} tag=${tag} status=FAIL log=${log_file}${NC}"
    ((FAIL+=1))
  fi

  echo
done

echo "=========================== SUMMARY ========================="
echo "OK=${OK}  FAIL=${FAIL}  SKIP=${SKIP}"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

# 失败则返回非0，方便 CI / bash 判断
if [[ "${FAIL}" -gt 0 ]]; then
  exit 2
fi
