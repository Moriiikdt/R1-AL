import argparse
import json
import re
from tqdm import tqdm
from pathlib import Path

def string_match(answer, prediction, choices):
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not prediction_tokens:
        return False

    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file to be evaluated')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f if line.strip()]

    # 总样本数（文件中所有行/样本）
    total_samples_all = len(input_data)

    corr, total_with_pred = 0, 0  # corr: 正确计数；total_with_pred: 有提取到 <answer> 的样本数

    modality_metrics = {
        'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0],
        'mix-sound-music': [0, 0], 'mix-sound-speech': [0, 0],
        'mix-music-speech': [0, 0], 'mix-sound-music-speech': [0, 0]
    }
    category_metrics = {
        'Signal Layer': [0, 0], 'Perception Layer': [0, 0],
        'Semantic Layer': [0, 0], 'Cultural Layer': [0, 0]
    }
    subcat_metrics = {}

    output_key = 'model_output'  # 模型输出字段
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    # 正则：提取 <answer>...</answer>（大小写不敏感，允许跨行）
    answer_pattern = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)

    # 用 tqdm 显示进度（如不需要可改回 enumerate(input_data)）
    for idx, sample in enumerate(tqdm(input_data)):
        # 如果没有预测字段，直接跳过（但仍计入总样本数 total_samples_all）
        if output_key not in sample:
            no_pred_count += 1
            new_data.append(sample)
            continue

        raw_pred = sample.get(output_key, "")

        # 尝试提取 <answer>...</answer>
        m = answer_pattern.search(raw_pred)
        if not m:
            no_pred_count += 1
            new_data.append(sample)
            continue  # 跳过该样本（不计入 total_with_pred）

        # 若成功提取
        extracted = m.group(1).strip()
        sample[output_key] = extracted  # 覆盖写回
        _prediction = extracted

        _answer = sample['answer']
        modality = sample['modality']
        # category = sample['category']
        category = "1"
        choices = sample['choices']

        subcat = sample.get('sub-category', None)
        if subcat is not None and subcat not in subcat_metrics:
            subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        # 增加“有提取到答案”的计数
        total_with_pred += 1

        if match_result:
            # 计数存在性保护（若遇到未预置的新键）
            if modality not in modality_metrics:
                modality_metrics[modality] = [0, 0]
            if category not in category_metrics:
                category_metrics[category] = [0, 0]

            modality_metrics[modality][0] += 1
            category_metrics[category][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        new_data.append(sample)

        # 无论是否匹配，增加该 modality/category 的样本计数
        if modality not in modality_metrics:
            modality_metrics[modality] = [0, 0]
        if category not in category_metrics:
            category_metrics[category] = [0, 0]

        modality_metrics[modality][1] += 1
        category_metrics[category][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    print("*" * 30)
    print("Modality-wise Accuracy (on samples that had extracted outputs):")
    for modality in modality_metrics:
        n_correct, n_total = modality_metrics[modality]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{modality} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Category-wise Accuracy (on samples that had extracted outputs):")
    for category in category_metrics:
        n_correct, n_total = category_metrics[category]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{category} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy (on samples that had extracted outputs):")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    # 主要变更：用总样本数（input_data 的长度）作为分母来计算“总体准确率”
    overall_acc_all = (corr / total_samples_all) * 100 if total_samples_all > 0 else 0.0
    # 同时保留原来的“只在有提取到答案的样本上计算”的准确率（以便比对）
    overall_acc_pred = (corr / total_with_pred) * 100 if total_with_pred > 0 else 0.0

    print(f"Total samples in file: {total_samples_all}")
    print(f"Samples with extracted <answer>: {total_with_pred}")
    print(f"Correct answers (corr): {corr}")
    print(f"Overall Accuracy (corr / total_samples_all): {overall_acc_all:.2f}%")
    print(f"Accuracy on predicted samples (corr / samples_with_extracted_answer): {overall_acc_pred:.2f}%")
    print("*" * 30)
    print(f"No prediction (no <answer> extracted) count: {no_pred_count}")
