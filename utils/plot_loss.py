#!/usr/bin/env python3
# plot_loss_from_logs.py
import re
import argparse
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    epochs = []
    losses = []
    pattern = re.compile(r'\[Epoch\]\s+(\d+).*?loss_overall\s+([\d\.]+)')
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    # Сортируем по эпохам на случай, если строки не по порядку
    if epochs:
        pairs = sorted(zip(epochs, losses))
        epochs, losses = zip(*pairs)
    return list(epochs), list(losses)

def main():
    parser = argparse.ArgumentParser(description='Plot train/val loss from .log files')
    parser.add_argument('--train_log', required=True, help='Path to train .log file')
    parser.add_argument('--val_log', required=True, help='Path to validation .log file')
    parser.add_argument('--save_path', default='loss_overall.png', help='Output image path')
    parser.add_argument('--title', default='Loss Overall', help='Plot title')
    args = parser.parse_args()

    train_epochs, train_losses = parse_log_file(args.train_log)
    val_epochs, val_losses = parse_log_file(args.val_log)

    if not train_epochs:
        print("Warning: No training data found!")
    if not val_epochs:
        print("Warning: No validation data found!")

    plt.figure(figsize=(10, 6))
    if train_epochs:
        plt.plot(train_epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    if val_epochs:
        plt.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Overall')
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.show()
    print(f"Plot saved to {args.save_path}")

if __name__ == '__main__':
    main()
