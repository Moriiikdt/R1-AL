#!/usr/bin/env python3
# save as: extract_eval_metrics.py

import os
import re
import sys
from pathlib import Path

LOG_DIR = Path("/mnt/hdfs/if_au/saves/mrx/results/jsonl/_eval_logs")
MIN_TOTAL = 1000

# regex patterns
re_total = re.compile(r"Total samples in file:\s*(\d+)")
re_task_line = re.compile(r"^(sound|music|speech)\s*:\s*([0-9.]+%)\s*over\s*(\d+)\s*samples\s*$")
re_acc_pred = re.compile(
    r"^Accuracy on predicted samples\s*\(corr\s*/\s*samples_with_extracted_answer\)\s*:\s*([0-9.]+%)\s*$"
)

def parse_log(text: str):
    """
    Return dict with:
      total (int or None),
      sound/music/speech -> (pct, n) or None,
      acc_pred (pct str) or None
    Use last occurrence if multiple.
    """
    totals = re_total.findall(text)
    total = int(totals[-1]) if totals else None

    sound = music = speech = None
    acc_pred = None

    # scan line by line to take last matches
    for line in text.splitlines():
        line_stripped = line.strip()

        m1 = re_task_line.match(line_stripped)
        if m1:
            k, pct, n = m1.group(1), m1.group(2), int(m1.group(3))
            if k == "sound":
                sound = (pct, n)
            elif k == "music":
                music = (pct, n)
            elif k == "speech":
                speech = (pct, n)
            continue

        m2 = re_acc_pred.match(line_stripped)
        if m2:
            acc_pred = m2.group(1)

    return {
        "total": total,
        "sound": sound,
        "music": music,
        "speech": speech,
        "acc_pred": acc_pred,
    }

def fmt_task(name, val):
    if val is None:
        return f"{name} : N/A"
    pct, n = val
    return f"{name} : {pct} over {n} samples"

def main():
    log_dir = LOG_DIR
    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])

    if not log_dir.exists():
        print(f"[ERROR] log dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    logs = sorted([p for p in log_dir.iterdir() if p.is_file() and p.suffix == ".log"])
    if not logs:
        print(f"[WARN] no .log files under {log_dir}")
        return

    any_printed = False

    for p in logs:
        try:
            text = p.read_text(errors="ignore")
        except Exception as e:
            print(f"[SKIP] cannot read {p.name}: {e}", file=sys.stderr)
            continue

        info = parse_log(text)
        total = info["total"]

        # only output if total >= MIN_TOTAL
        if total is None or total < MIN_TOTAL:
            continue

        any_printed = True
        print("============================================================")
        print(f"[FILE] {p.name}")
        print(f"[INFO] Total samples in file: {total}")
        print(fmt_task("sound", info["sound"]))
        print(fmt_task("music", info["music"]))
        print(fmt_task("speech", info["speech"]))
        print(f"Accuracy on predicted samples (corr / samples_with_extracted_answer): {info['acc_pred'] or 'N/A'}")

    if not any_printed:
        print(f"[INFO] No logs matched condition: Total samples in file >= {MIN_TOTAL}")

if __name__ == "__main__":
    main()
