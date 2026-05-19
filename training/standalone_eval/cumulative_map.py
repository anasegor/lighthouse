# compute_cumulative_map.py
import json
import argparse
import numpy as np
from standalone_eval.utils import load_jsonl
from standalone_eval.eval import (
    compute_mr_ap,
    get_data_by_range,
)  # предполагаем, что ваш скрипт eval лежит рядом

def compute_cumulative_map(submission, ground_truth, max_len=None, step=1.0):
    """
    submission, ground_truth – списки словарей (формат QVHighlight)
    max_len – максимальная длина (если None – берётся из данных)
    step – шаг перебора длины (сек)
    Возвращает:
        lengths: np.array длин порогов
        cum_map: np.array значений mAP (средний по IoU)
    """
    # Определяем диапазон длин (половина длительности окна – width)
    all_gt_lengths = []
    for d in ground_truth:
        for w in d["relevant_windows"]:
            all_gt_lengths.append(w[1] - w[0])
    min_len = 0.0
    max_len = max_len or np.max(all_gt_lengths)

    thresholds = np.arange(min_len, max_len + step, step)
    cum_map = []

    for L in thresholds:
        # фильтруем GT и submission, оставляя только моменты длиной <= L
        sub_in, gt_in = get_data_by_range(submission, ground_truth, [0, L])
        if len(gt_in) == 0:
            cum_map.append(0.0)
            continue
        # считаем mAP (нам нужно среднее по IoU порогам)
        iou_thd2ap = compute_mr_ap(sub_in, gt_in, num_workers=4) 
        avg_map = iou_thd2ap.get("average", 0.0)
        cum_map.append(avg_map)

    return np.array(thresholds), np.array(cum_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--save_path", default="cumulative_map.json")
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()

    submission = load_jsonl(args.submission_path)
    ground_truth = load_jsonl(args.gt_path)

    lengths, cum_map = compute_cumulative_map(
        submission, ground_truth, step=args.step
    )

    # Сохраняем кривую
    result = {"lengths": lengths.tolist(), "cumulative_mAP": cum_map.tolist()}
    with open(args.save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Cumulative mAP saved to {args.save_path}")

if __name__ == "__main__":
    main()