"""Plot Fig. 3b parity panels for scratch and pretrained MEGNET models."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'y_DFT', 'y_pred_scratch', 'y_pred_pretrained'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Input CSV must contain {sorted(required)}')
        rows = list(reader)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input CSV is empty or contains duplicate IDs')
    values = np.array([[float(row['y_DFT']), float(row['y_pred_scratch']),
                        float(row['y_pred_pretrained'])] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError('DFT and prediction values must be finite')
    return values


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / denominator
    return mae, r2


def plot(values):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 12})
    y_true = values[:, 0]
    predictions = [('MEGNET(F)', values[:, 1]), ('MEGNET(P)', values[:, 2])]
    extrema = np.concatenate([values[:, 0], values[:, 1], values[:, 2]])
    padding = (extrema.max() - extrema.min()) * 0.08
    limits = (extrema.min() - padding, extrema.max() + padding)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    for ax, (title, y_pred) in zip(axes, predictions):
        mae, r2 = metrics(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidths=0.35,
                   s=25, color='#1f77b4')
        ax.plot(limits, limits, color='#d62728', linestyle='--', linewidth=2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=28, pad=12)
        ax.set_xlabel('DFT formation enthalpy (meV/atom)', fontsize=21)
        ax.tick_params(which='both', direction='in', labelsize=15, width=1.5)
        ax.text(0.05, 0.94, f'$R^2$ = {r2:.2f}\nMAE = {mae:.1f} meV/atom',
                transform=ax.transAxes, va='top', fontsize=18,
                bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'})
    axes[0].set_ylabel('Predicted formation enthalpy (meV/atom)', fontsize=21)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=REPO / 'source_data/Source_Data_Fig3b_3d.csv')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/fig3/Fig3b.png')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    values = load_data(args.input)
    for name, column in [('scratch', 1), ('pretrained', 2)]:
        mae, r2 = metrics(values[:, 0], values[:, column])
        print(f'{name}: MAE={mae:.6f} meV/atom, R2={r2:.6f}')
    fig = plot(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
