# find_thresholds.py
import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema

def find_inflection_points(lengths, cum_map, smooth_sigma=2):
    """
    Находит индексы точек перегиба (локальные минимумы второй производной).
    Возвращает lengths этих точек.
    """
    # Сглаживаем кривую для устойчивости
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(cum_map, sigma=smooth_sigma)
    
    # Первая и вторая производная
    first_deriv = np.gradient(smoothed, lengths)
    second_deriv = np.gradient(first_deriv, lengths)
    
    # Ищем индексы локальных минимумов второй производной (максимальная кривизна)
    inflection_idx = argrelextrema(second_deriv, np.less)[0]
    if len(inflection_idx) == 0:
        # fallback: точки с минимальной второй производной
        inflection_idx = np.argsort(second_deriv)[:5]
    return lengths[inflection_idx]

def cluster_thresholds(inflection_lengths, n_clusters=2):
    """
    Кластеризует длины точек перегиба на n_clusters групп.
    Возвращает центры кластеров (границы между классами длин).
    """
    X = inflection_lengths.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten().tolist())
    return centers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative_map_path", required=True)
    parser.add_argument("--n_classes", type=int, default=3,
                        help="число классов длины (обычно 3 -> 2 границы)")
    parser.add_argument("--smooth_sigma", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.cumulative_map_path, "r") as f:
        data = json.load(f)
    lengths = np.array(data["lengths"])
    cum_map = np.array(data["cumulative_mAP"])

    # 1. Точки перегиба
    inflection_lengths = find_inflection_points(lengths, cum_map, args.smooth_sigma)
    print(f"Found {len(inflection_lengths)} inflection points at lengths: {inflection_lengths}")

    # 2. Кластеризация (получаем пороги, например для 3 классов нужно 2 границы)
    boundaries = cluster_thresholds(inflection_lengths, n_clusters=args.n_classes - 1)
    print(f"Length class boundaries: {boundaries}")

    # Сохраняем
    result = {
        "inflection_points": inflection_lengths.tolist(),
        "boundaries": boundaries,
    }
    with open("length_thresholds.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to length_thresholds.json")

if __name__ == "__main__":
    main()