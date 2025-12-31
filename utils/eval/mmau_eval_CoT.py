import argparse
import json
import re
from tqdm import tqdm

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

    corr, total_with_pred = 0, 0
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    subcat_metrics = {}
    output_key = 'model_output'
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    # 🔹 正则提取 <answer>...</answer>
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    for idx, sample in enumerate(tqdm(input_data)):
        if output_key not in sample:
            no_pred_count += 1
            new_data.append(sample)
            continue

        _prediction = sample.get(output_key, "")
        _answer = sample['answer']
        task = sample.get('task', 'unknown')
        difficulty = sample.get('difficulty', 'unknown')
        choices = sample['choices']
        subcat = sample.get('sub-category', None)

        # 🔹 提取标签内内容；如果没提取出来则跳过
        match = answer_pattern.search(_prediction)
        if not match:
            no_pred_count += 1
            new_data.append(sample)
            continue  # ⛔ 跳过这一条数据

        extracted_answer = match.group(1).strip()
        sample[output_key] = extracted_answer
        _prediction = extracted_answer  # 覆盖预测值

        if subcat is not None:
            subcat_metrics.setdefault(subcat, [0, 0])

        match_result = string_match(_answer, _prediction, choices)

        # 增加“有提取到答案”的计数
        total_with_pred += 1

        if match_result:
            task_metrics.setdefault(task, [0, 0])
            diff_metrics.setdefault(difficulty, [0, 0])

            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        new_data.append(sample)
        task_metrics.setdefault(task, [0, 0])
        diff_metrics.setdefault(difficulty, [0, 0])

        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    print("*" * 30)
    print("Task-wise Accuracy (on samples that had extracted outputs):")
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{task} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Difficulty-wise Accuracy (on samples that had extracted outputs):")
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{diff} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy (on samples that had extracted outputs):")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    # 主要变更：用总样本数（input_data 的长度）作为主要分母来计算“总体准确率”
    overall_acc_all = (corr / total_samples_all) * 100 if total_samples_all > 0 else 0.0
    overall_acc_pred = (corr / total_with_pred) * 100 if total_with_pred > 0 else 0.0

    print(f"Total samples in file: {total_samples_all}")
    print(f"Samples with extracted <answer>: {total_with_pred}")
    print(f"Correct answers (corr): {corr}")
    print(f"Overall Accuracy (corr / total_samples_all): {overall_acc_all:.2f}%")
    print(f"Accuracy on predicted samples (corr / samples_with_extracted_answer): {overall_acc_pred:.2f}%")
    print("*" * 30)
    print(f"No prediction (no <answer> extracted) count: {no_pred_count}")
