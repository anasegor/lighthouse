# find_thresholds.py (обновлённый)
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

def find_inflection_points(lengths, cum_map, smooth_sigma=2):
    smoothed = gaussian_filter1d(cum_map, sigma=smooth_sigma)
    first_deriv = np.gradient(smoothed, lengths)
    second_deriv = np.gradient(first_deriv, lengths)
    inflection_idx = argrelextrema(second_deriv, np.less)[0]
    if len(inflection_idx) == 0:
        inflection_idx = np.argsort(second_deriv)[:5]
    return lengths[inflection_idx]

def cluster_thresholds(inflection_lengths, n_clusters=2):
    X = inflection_lengths.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten().tolist())
    return centers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative_map_path", required=True)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--smooth_sigma", type=float, default=2.0)
    parser.add_argument("--max_a_l", type=float, default=None,
                        help="Maximum video length (e.g., 75). If provided, boundaries will be normalized by this value.")
    parser.add_argument("--save_path", default="length_thresholds.json")
    args = parser.parse_args()

    with open(args.cumulative_map_path, "r") as f:
        data = json.load(f)
    lengths = np.array(data["lengths"])
    cum_map = np.array(data["cumulative_mAP"])

    # 1. Поиск точек перегиба (в единицах lengths – абсолютных)
    inflection_lengths = find_inflection_points(lengths, cum_map, args.smooth_sigma)
    print(f"Found {len(inflection_lengths)} inflection points at lengths: {inflection_lengths}")

    # 2. Кластеризация для получения границ
    boundaries_abs = cluster_thresholds(inflection_lengths, n_clusters=args.n_classes - 1)

    # 3. Нормализация (если задан max_a_l)
    if args.max_a_l is not None:
        inflection_lengths_norm = (inflection_lengths / args.max_a_l).tolist()
        boundaries_norm = [round(b / args.max_a_l, 4) for b in boundaries_abs]
        print(f"Normalized boundaries (range 0-1): {boundaries_norm}")
    else:
        inflection_lengths_norm = inflection_lengths.tolist()
        boundaries_norm = boundaries_abs

    result = {
        "inflection_points": inflection_lengths_norm,
        "boundaries": boundaries_norm,
        "max_a_l": args.max_a_l
    }
    with open(args.save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.save_path}")

if __name__ == "__main__":
    main()