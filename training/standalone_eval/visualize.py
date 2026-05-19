# visualize_cumulative_map.py
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

def find_inflection_points(lengths, cum_map, smooth_sigma=2):
    """Возвращает индексы и длины точек перегиба."""
    smoothed = gaussian_filter1d(cum_map, sigma=smooth_sigma)
    first_deriv = np.gradient(smoothed, lengths)
    second_deriv = np.gradient(first_deriv, lengths)
    inflection_idx = argrelextrema(second_deriv, np.less)[0]
    if len(inflection_idx) == 0:
        # fallback: 5 точек с самой отрицательной второй производной
        inflection_idx = np.argsort(second_deriv)[:5]
    return inflection_idx, lengths[inflection_idx]

def plot_cumulative_map(lengths, cum_map, inflection_lengths=None, boundaries=None, save_path="cumulative_map_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lengths, cum_map, 'b-', linewidth=2, label='Cumulative mAP')
    ax.fill_between(lengths, cum_map, alpha=0.1, color='blue')
    
    if inflection_lengths is not None and len(inflection_lengths) > 0:
        ax.scatter(inflection_lengths, 
                   np.interp(inflection_lengths, lengths, cum_map),
                   color='red', s=80, zorder=5, label='Inflection points')
    
    if boundaries is not None:
        for b in boundaries:
            ax.axvline(x=b, color='green', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Boundary {b:.1f}')
    
    ax.set_xlabel('Moment length (s)')
    ax.set_ylabel('Cumulative mAP (average over IoU)')
    ax.set_title('Cumulative mAP vs. Moment Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

def plot_clusters(inflection_lengths, boundaries=None, save_path="clusters_plot.png"):
    if len(inflection_lengths) == 0:
        print("No inflection points to cluster.")
        return
    
    # Если есть границы, окрашиваем точки по кластерам
    if boundaries is not None:
        boundaries = sorted(boundaries)
        clusters = np.digitize(inflection_lengths, boundaries)  # 0,1,2 для 2 границ
        cmap = plt.cm.Set1
    else:
        clusters = np.zeros(len(inflection_lengths))
        cmap = plt.cm.Greys
    
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(inflection_lengths, np.zeros_like(inflection_lengths),
                         c=clusters, cmap=cmap, s=100, edgecolors='k',
                         alpha=0.8, label='Inflection points')
    
    if boundaries is not None:
        for i, b in enumerate(boundaries):
            ax.axvline(x=b, color='green', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Boundary {b:.1f}')
        # Добавляем легенду цветов
        legend1 = ax.legend(loc='upper left')
        ax.legend(*scatter.legend_elements(), title="Clusters", loc='upper right')
        ax.add_artist(legend1)
    
    ax.set_xlabel('Moment length (s)')
    ax.set_yticks([])
    ax.set_title('Inflection Points Clustering')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative_map_path", default="cumulative_map.json",
                        help="JSON с lengths и cumulative_mAP")
    parser.add_argument("--thresholds_path", default="length_thresholds.json",
                        help="JSON с inflection_points и boundaries (опционально)")
    parser.add_argument("--smooth_sigma", type=float, default=2.0,
                        help="Степень сглаживания для поиска перегибов")
    args = parser.parse_args()

    set_style()
    
    with open(args.cumulative_map_path, 'r') as f:
        data = json.load(f)
    lengths = np.array(data['lengths'])
    cum_map = np.array(data['cumulative_mAP'])
    
    # Ищем inflection points (чтобы нарисовать, даже если thresholds_path нет)
    inflection_idx, inflection_lengths = find_inflection_points(lengths, cum_map, args.smooth_sigma)
    print(f"Found {len(inflection_lengths)} inflection points at lengths: {inflection_lengths}")
    
    boundaries = None
    if args.thresholds_path:
        try:
            with open(args.thresholds_path, 'r') as f:
                thr_data = json.load(f)
            inflection_lengths = np.array(thr_data.get('inflection_points', inflection_lengths))
            boundaries = thr_data.get('boundaries', None)
            print(f"Loaded boundaries: {boundaries}")
        except FileNotFoundError:
            print("Thresholds file not found, will compute inflection points manually.")
    
    # Построение графиков
    plot_cumulative_map(lengths, cum_map, inflection_lengths, boundaries,
                        save_path="cumulative_map_plot.png")
    
    if len(inflection_lengths) > 0:
        plot_clusters(inflection_lengths, boundaries,
                      save_path="clusters_plot.png")

if __name__ == "__main__":
    main()