"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

MIT License

Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp
from standalone_eval.utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    qid2target_vid = {d["qid"]: d["vid"] for d in ground_truth}
    pred_qid2data = defaultdict(list)
    for d in submission:
        qid = d["qid"]
        if d.get("vid") != qid2target_vid.get(qid, None):
            continue
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    data_triples = []
    for qid, gt_data in gt_qid2data.items():
        if qid in pred_qid2data and len(pred_qid2data[qid]) > 0:
            data_triples.append([qid, gt_data, pred_qid2data[qid]])

    if len(data_triples) == 0:
        ap_thds = np.zeros(len(iou_thds))
        iou_thd2ap = {str(thd): 0.0 for thd in iou_thds}
        iou_thd2ap["average"] = 0.0
        return {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    ap_array = np.array(list(qid2ap_list.values()))
    ap_thds = ap_array.mean(0)
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    qid2target_vid = {d["qid"]: d["vid"] for d in ground_truth}
    qid2pred_window = {}
    for d in submission:
        qid = d["qid"]
        if d.get("vid") != qid2target_vid.get(qid, None):
            continue
        qid2pred_window[qid] = d["pred_relevant_windows"][0][:2]

    qid2gt_window = {}
    for d in ground_truth:
        qid = d["qid"]
        cur_gt_windows = d["relevant_windows"]
        if len(cur_gt_windows) == 0:
            qid2gt_window[qid] = [0.0, 0.0]
            continue
        if qid in qid2pred_window:
            pred_win = qid2pred_window[qid]
            ious = compute_temporal_iou_batch_cross(
                np.array([pred_win]), np.array(cur_gt_windows)
            )[0]
            best_idx = np.argmax(ious)
            qid2gt_window[qid] = cur_gt_windows[best_idx]
        else:
            qid2gt_window[qid] = cur_gt_windows[0]

    qids = [qid for qid in qid2pred_window.keys() if qid in qid2gt_window]
    if len(qids) == 0:
        return {str(thd): 0.0 for thd in iou_thds}

    pred_windows = np.array([qid2pred_window[qid] for qid in qids]).astype(float)
    gt_windows = np.array([qid2gt_window[qid] for qid in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)

    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{100 * np.mean(pred_gt_iou >= thd):.2f}")
    return iou_thd2recall_at_one


def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range):
    min_l, max_l = len_range
    if min_l == 0 and max_l == 150:
        return submission, ground_truth

    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [
            w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    # Определяем диапазоны длительности (в секундах)
    length_ranges = [
        [0, 1500],   # full
        [0, 10], [10, 20], [20, 30], [30, 40], [40, 50],
        [50, 60], [60, 70], [70, 80], [80, 90], [90, 1500]
    ]
    range_names = [
        "full", "0-10", "11-20", "21-30", "31-40", "41-50",
        "51-60", "61-70", "71-80", "81-90", "90+"
    ]

    ret_metrics = {}
    iou_thds = np.linspace(0.5, 0.95, 10)  # для заполнения нулями

    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")

        # Если после фильтрации не осталось данных, заполняем нулями
        if len(_ground_truth) == 0:
            iou_thd2ap = {str(thd): 0.0 for thd in iou_thds}
            iou_thd2ap["average"] = 0.0
            iou_thd2recall = {str(thd): 0.0 for thd in iou_thds}
            ret_metrics[name] = {"MR-mAP": iou_thd2ap, "MR-R1": iou_thd2recall}
            if verbose:
                print(f"[eval_moment_retrieval] [{name}] no data, skipped")
            continue

        iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
        iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth)
        ret_metrics[name] = {"MR-mAP": iou_thd2average_precision, "MR-R1": iou_thd2recall_at_one}
        if verbose:
            print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {k: np.argmax(v["pred_saliency_scores"]) for k, v in qid2preds.items()}
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {k: v["pred_saliency_scores"] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, w_idx, score in pool.imap_unordered(
                    compute_ap_from_tuple, input_tuples, chunksize=chunksize):
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score

    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        y_predict = y_predict[:len(y_true)]
    elif len(y_true) > len(y_predict):
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


def mk_gt_scores(gt_data, clip_length=2):
    num_clips = int(gt_data["duration"] / clip_length)
    saliency_scores_full_video = np.zeros((num_clips, 3))
    relevant_clip_ids = np.array(gt_data["relevant_clip_ids"])
    saliency_scores_relevant_clips = np.array(gt_data["saliency_scores"])
    saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    return saliency_scores_full_video


def eval_highlight(submission, ground_truth, verbose=True):
    qid2preds = {d["qid"]: d for d in submission}
    qid2gt_scores_full_range = {d["qid"]: mk_gt_scores(d) for d in ground_truth}
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ["Fair", "Good", "VeryGood"]
    highlight_det_metrics = {}
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list, saliency_score_names):
        start_time = time.time()
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full_range.items()}
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f"HL-min-{score_name}"] = {"HL-mAP": mean_ap, "HL-Hit1": hit_at_one}
        if verbose:
            print(f"Calculating highlight scores with min score {gt_saliency_score_min} ({score_name})")
            print(f"Time cost {time.time() - start_time:.2f} seconds")
    return highlight_det_metrics


def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()

    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)

        # Добавляем краткие метрики для каждого диапазона длительности
        for range_name, range_scores in moment_ret_scores.items():
            mrmap = range_scores["MR-mAP"]
            mr_r1 = range_scores["MR-R1"]
            eval_metrics_brief[f"MR-{range_name}-mAP"] = mrmap["average"]
            eval_metrics_brief[f"MR-{range_name}-mAP@0.5"] = mrmap["0.5"]
            eval_metrics_brief[f"MR-{range_name}-mAP@0.75"] = mrmap["0.75"]
            eval_metrics_brief[f"MR-{range_name}-R1@0.5"] = mr_r1["0.5"]
            eval_metrics_brief[f"MR-{range_name}-R1@0.7"] = mr_r1["0.7"]

    if "pred_saliency_scores" in submission[0]:
        highlight_det_scores = eval_highlight(submission, ground_truth, verbose=verbose)
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict([
            (f"{k}-{sub_k.split('-')[1]}", v[sub_k])
            for k, v in highlight_det_scores.items() for sub_k in v])
        eval_metrics_brief.update(sorted(highlight_det_scores_brief.items(), key=lambda x: x[0]))

    # Сортируем все краткие метрики по ключу
    eval_metrics_brief = OrderedDict(sorted(eval_metrics_brief.items(), key=lambda x: x[0]))

    # Сортируем полные метрики по ключу
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
