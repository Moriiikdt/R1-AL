import json
from pathlib import Path

# ===== 输入输出 =====
input_json = Path(
    "../data/avqa_RL_46544_format.json"
)
output_json = Path(
    "../data/avqa_RL_46544_format_path.json"
)

# JSON 文件所在目录
base_dir = input_json.parent

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

converted_cnt = 0

for item in data:
    if "audios" not in item:
        continue

    new_audios = []
    for audio in item["audios"]:
        audio = audio.strip()

        # ✅ 核心逻辑：
        # 把 ../audio/xxx.wav 视为 ./audio/xxx.wav
        if audio.startswith("../audio/"):
            audio = audio.replace("../audio/", "audio/", 1)

        # 拼成绝对路径
        abs_path = (base_dir / audio).resolve()
        new_audios.append(str(abs_path))
        converted_cnt += 1

    item["audios"] = new_audios

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("===================================")
print(f"Input file   : {input_json}")
print(f"Output file  : {output_json}")
print(f"Converted audios: {converted_cnt}")
print("===================================")
