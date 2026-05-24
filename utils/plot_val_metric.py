#!/usr/bin/env python3
# plot_val_metric.py
import re
import json
import argparse
import matplotlib.pyplot as plt


def parse_metric_from_log(log_path, metric_key="MR-full-mAP"):
    epochs = []
    values = []
    pattern = re.compile(r"\[Epoch\]\s+(\d+).*?\[Metrics\]\s+(\{.*\})")
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            try:
                metrics = json.loads(match.group(2))
                value = metrics["brief"][metric_key]
            except (KeyError, json.JSONDecodeError) as e:
                print(
                    f"Warning: could not parse metrics in {log_path} epoch {epoch}: {e}"
                )
                continue
            epochs.append(epoch)
            values.append(value)
    # Сортируем по эпохам
    if epochs:
        pairs = sorted(zip(epochs, values))
        epochs, values = zip(*pairs)
    return list(epochs), list(values)


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation MR-full-mAP from one or more val.log files"
    )
    parser.add_argument(
        "--val_logs",
        nargs="+",
        required=True,
        help="Paths to val.log files (one or more)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom labels for each log file (same order as val_logs)",
    )
    parser.add_argument(
        "--metric_key",
        default="MR-full-mAP",
        help="Key inside 'brief' metrics dict (default: MR-full-mAP)",
    )
    parser.add_argument(
        "--save_path", default="val_metric.png", help="Output image path"
    )
    parser.add_argument(
        "--title", default="Validation MR-full-mAP on Castella", help="Plot title"
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.val_logs):
        parser.error("Number of labels must match number of log files")

    plt.figure(figsize=(10, 6))
    for i, log_path in enumerate(args.val_logs):
        epochs, values = parse_metric_from_log(log_path, args.metric_key)
        if not epochs:
            print(f"Warning: No metric data found in {log_path}")
            continue
        label = args.labels[i] if args.labels else log_path.split("/")[-1]  # имя файла
        plt.plot(epochs, values, linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(args.metric_key)
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()
    print(f"Plot saved to {args.save_path}")


if __name__ == "__main__":
    main()
