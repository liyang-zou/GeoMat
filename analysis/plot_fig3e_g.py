"""Plot the task-specific cumulative recall curves in Fig. 3e-g."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


REPO = Path(__file__).resolve().parents[1]
TASKS = {
    'afm': {
        'panel': 'Fig3e',
        'source': 'Source_Data_Fig3e.csv',
        'threshold': 400.0,
        'criterion': r'$T_N > 400$ K',
        'title': r'Cumulative recall of high-$T_N$ materials',
    },
    'exciton': {
        'panel': 'Fig3f',
        'source': 'Source_Data_Fig3f.csv',
        'threshold': 1.94,
        'criterion': r'$E_b > 1.94$ eV',
        'title': r'Cumulative recall of high-$E_b$ materials',
    },
    'tc8055': {
        'panel': 'Fig3g',
        'source': 'Source_Data_Fig3g.csv',
        'threshold': 5.0,
        'criterion': r'$T_c > 5$ K',
        'title': r'Cumulative recall of superconductors',
    },
}


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized not in {'true', 'false'}:
        raise ValueError(f'Invalid Boolean value: {value}')
    return normalized == 'true'


def load_ranked(path, threshold):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'y_true', 'y_pred', 'is_target'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Input CSV must contain {sorted(required)}')
        rows = list(reader)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input CSV is empty or contains duplicate IDs')
    y_true = np.array([float(row['y_true']) for row in rows])
    y_pred = np.array([float(row['y_pred']) for row in rows])
    recorded_targets = np.array([parse_bool(row['is_target']) for row in rows])
    expected_targets = y_true > threshold
    if not np.array_equal(recorded_targets, expected_targets):
        raise ValueError('is_target does not match y_true and the task threshold')
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('Ground-truth and prediction values must be finite')
    order = np.argsort(-y_pred, kind='stable')
    return expected_targets[order]


def recall_curve(ranked_targets):
    total_targets = int(ranked_targets.sum())
    if total_targets == 0:
        raise ValueError('No material satisfies the task criterion')
    n_samples = len(ranked_targets)
    fraction = np.arange(1, n_samples + 1) / n_samples
    recall = np.cumsum(ranked_targets) / total_targets
    return (np.insert(fraction, 0, 0.0),
            np.insert(recall, 0, 0.0), total_targets)


def plot(x, recall, screen_fraction, title):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12,
    })
    n_samples = len(x) - 1
    n_screened = max(1, int(np.ceil(n_samples * screen_fraction)))
    actual_fraction = n_screened / n_samples
    recall_at_fraction = recall[n_screened]

    fig, ax = plt.subplots(figsize=(13.7, 13.7))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=8,
            label='Random Guess')
    ax.step(x, recall, where='post', color='#d62728', linewidth=8,
            label='MEGNET(P) Prediction')
    ax.scatter(actual_fraction, recall_at_fraction, color='#1f77b4',
               s=400, zorder=10)
    ax.annotate(f'Recall @ {screen_fraction:.0%}: {recall_at_fraction:.1%}',
                xy=(actual_fraction, recall_at_fraction),
                xytext=(actual_fraction + 0.02,
                        max(0.04, recall_at_fraction - 0.09)),
                color='#1f77b4', fontsize=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Fraction of Candidates Screened', fontsize=45, labelpad=20)
    ax.set_ylabel('Fraction of Target Materials Found', fontsize=45, labelpad=20)
    ax.set_title(title, fontsize=50, pad=25)
    ax.tick_params(which='major', direction='in', labelsize=40,
                   width=2, length=6, pad=8)
    ax.tick_params(which='minor', direction='in', width=1.5, length=3)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: '' if np.isclose(value, 0.0)
                      else f'{value:.1f}'))
    ax.legend(fontsize=35)
    fig.tight_layout()
    return fig, n_screened, recall_at_fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=tuple(TASKS))
    parser.add_argument('--input', type=Path,
                        help='Override the task-specific Source Data CSV')
    parser.add_argument('--output', type=Path,
                        help='Override the task-specific output image')
    parser.add_argument('--screen-fraction', type=float, default=0.20)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if not 0 < args.screen_fraction <= 1:
        parser.error('--screen-fraction must be in (0, 1]')

    config = TASKS[args.task]
    input_path = args.input or REPO / 'source_data' / config['source']
    output_path = args.output or REPO / 'outputs' / 'fig3' / f"{config['panel']}.png"
    ranked = load_ranked(input_path, config['threshold'])
    x, recall, total_targets = recall_curve(ranked)
    fig, n_screened, recall_at_fraction = plot(
        x, recall, args.screen_fraction, config['title'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"{args.task}: {total_targets} targets ({config['criterion']}) "
          f"among {len(ranked)} test materials")
    print(f'Recall after screening {n_screened}/{len(ranked)} candidates: '
          f'{recall_at_fraction:.6%}')
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
