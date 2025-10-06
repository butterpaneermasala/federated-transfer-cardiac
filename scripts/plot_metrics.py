#!/usr/bin/env python3
"""
Plot per-round metrics from client agents.
Reads CSVs from reports/metrics_<hospital_id>.csv and generates plots:
- reports/metrics_accuracy.png
- reports/metrics_loss.png

Usage:
  PYTHONPATH=src python scripts/plot_metrics.py
"""
import os
import glob
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(REPO_ROOT, 'reports')


def load_metrics():
    data = defaultdict(lambda: {'round': [], 'loss': [], 'acc': []})
    for path in glob.glob(os.path.join(REPORTS_DIR, 'metrics_*.csv')):
        hospital_id = os.path.basename(path).replace('metrics_', '').replace('.csv', '')
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rnd = int(row['round'])
                    loss = float(row['loss'])
                    acc = float(row['accuracy'])
                except Exception:
                    continue
                data[hospital_id]['round'].append(rnd)
                data[hospital_id]['loss'].append(loss)
                data[hospital_id]['acc'].append(acc)
        # sort by round
        if data[hospital_id]['round']:
            order = sorted(range(len(data[hospital_id]['round'])), key=lambda i: data[hospital_id]['round'][i])
            data[hospital_id]['round'] = [data[hospital_id]['round'][i] for i in order]
            data[hospital_id]['loss'] = [data[hospital_id]['loss'][i] for i in order]
            data[hospital_id]['acc'] = [data[hospital_id]['acc'][i] for i in order]
    return data


def plot_metrics(data):
    if not data:
        print('No metrics CSV files found in reports/.')
        return

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    for hid, d in data.items():
        plt.plot(d['round'], d['acc'], marker='o', label=hid)
    plt.xlabel('Global Round')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Per-Hospital Training Accuracy by Round')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_acc = os.path.join(REPORTS_DIR, 'metrics_accuracy.png')
    plt.tight_layout()
    plt.savefig(out_acc, dpi=200)
    print(f'Saved: {out_acc}')
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    for hid, d in data.items():
        plt.plot(d['round'], d['loss'], marker='s', label=hid)
    plt.xlabel('Global Round')
    plt.ylabel('Average Training Loss')
    plt.title('Per-Hospital Training Loss by Round')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_loss = os.path.join(REPORTS_DIR, 'metrics_loss.png')
    plt.tight_layout()
    plt.savefig(out_loss, dpi=200)
    print(f'Saved: {out_loss}')
    plt.close()


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    data = load_metrics()
    plot_metrics(data)


if __name__ == '__main__':
    main()
