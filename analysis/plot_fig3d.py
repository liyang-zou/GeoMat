"""Plot Fig. 3d cumulative recall for DFT-stable Type-4 MAX structures."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized not in {'true', 'false'}:
        raise ValueError(f'Invalid Boolean value: {value}')
    return normalized == 'true'


def load_ranked(path, prediction_column, threshold):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'y_DFT', prediction_column, 'is_stable_DFT'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Input CSV must contain {sorted(required)}')
        rows = list(reader)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input CSV is empty or contains duplicate IDs')
    y_dft = np.array([float(row['y_DFT']) for row in rows])
    predictions = np.array([float(row[prediction_column]) for row in rows])
    recorded = np.array([parse_bool(row['is_stable_DFT']) for row in rows])
    stable = y_dft < threshold
    if not np.array_equal(recorded, stable):
        raise ValueError('is_stable_DFT does not match y_DFT and the requested threshold')
    if not np.isfinite(y_dft).all() or not np.isfinite(predictions).all():
        raise ValueError('DFT and prediction values must be finite')
    order = np.argsort(predictions, kind='stable')
    return stable[order]


def recall_curve(ranked_targets):
    total_targets = int(ranked_targets.sum())
    if total_targets == 0:
        raise ValueError('No DFT-stable structures satisfy the threshold')
    n = len(ranked_targets)
    fraction = np.arange(1, n + 1) / n
    recall = np.cumsum(ranked_targets) / total_targets
    return np.insert(fraction, 0, 0.0), np.insert(recall, 0, 0.0)


def plot(x, recall, screen_fraction):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 12})
    n = len(x) - 1
    n_screened = max(1, int(np.ceil(n * screen_fraction)))
    actual_fraction = n_screened / n
    recall_at_fraction = recall[n_screened]
    fig, ax = plt.subplots(figsize=(13.7, 13.7))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=8,
            label='Random Guess')
    ax.step(x, recall, where='post', color='#d62728', linewidth=8,
            label='MEGNET(P) Prediction')
    ax.scatter(actual_fraction, recall_at_fraction, color='#1f77b4', s=400, zorder=10)
    ax.annotate(f'Recall @ {screen_fraction:.0%}: {recall_at_fraction:.1%}',
                xy=(actual_fraction, recall_at_fraction),
                xytext=(actual_fraction + 0.02, recall_at_fraction - 0.08),
                color='#1f77b4', fontsize=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Fraction of Candidates Screened', fontsize=45, labelpad=20)
    ax.set_ylabel('Fraction of Target Materials Found', fontsize=45, labelpad=20)
    ax.set_title('Cumulative recall of stable structures', fontsize=50, pad=25)
    ax.tick_params(which='major', direction='in', labelsize=40, width=2, length=6)
    ax.tick_params(which='minor', direction='in', width=1.5, length=3)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: '' if np.isclose(value, 0.0)
                      else f'{value:.1f}'))
    ax.legend(fontsize=35)
    fig.tight_layout()
    return fig, n_screened, recall_at_fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=REPO / 'source_data/Source_Data_Fig3b_3d.csv')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/fig3/Fig3d.png')
    parser.add_argument('--prediction-column', default='y_pred_pretrained',
                        choices=('y_pred_pretrained', 'y_pred_scratch'))
    parser.add_argument('--threshold', type=float, default=30.0,
                        help='DFT stability threshold in meV/atom')
    parser.add_argument('--screen-fraction', type=float, default=0.05)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if not 0 < args.screen_fraction <= 1:
        parser.error('--screen-fraction must be in (0, 1]')
    ranked = load_ranked(args.input, args.prediction_column, args.threshold)
    x, recall = recall_curve(ranked)
    fig, n_screened, recall_at_fraction = plot(x, recall, args.screen_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'DFT targets below {args.threshold:g} meV/atom: {int(ranked.sum())}')
    print(f'Recall after screening {n_screened}/{len(ranked)} candidates: '
          f'{recall_at_fraction:.6%}')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
