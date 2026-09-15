"""Plot Fig. 2c relative to each task's smallest pretraining-corpus MAE."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'task', 'unit', 'pretrain_size', 'mae'}.issubset(reader.fieldnames or []):
            raise ValueError('Input CSV must contain task, unit, pretrain_size, and mae')
        rows = list(reader)
    if not rows:
        raise ValueError('Input CSV contains no observations')
    groups = {}
    for task in dict.fromkeys(row['task'] for row in rows):
        values = sorted((row for row in rows if row['task'] == task),
                        key=lambda row: int(row['pretrain_size']))
        sizes = np.array([int(row['pretrain_size']) for row in values])
        maes = np.array([float(row['mae']) for row in values])
        if (not np.isfinite(maes).all() or (sizes <= 0).any()
                or (maes <= 0).any() or len(set(sizes)) != len(sizes)):
            raise ValueError(f'{task}: invalid corpus sizes or MAE values')
        groups[task] = (sizes, (maes[0] - maes) / maes[0] * 100)
    return groups


def plot(groups):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 12})
    fig, ax = plt.subplots(figsize=(11.5, 12.6), constrained_layout=True)
    styles = {'JDFT2D': ('#1f77b4', 'o'), 'GVRH': ('#ff7f0e', 's'),
              'MP_E_Form': ('#d62728', '^')}
    for task, (sizes, relative) in groups.items():
        color, marker = styles.get(task, ('#333333', 'o'))
        ax.plot(sizes, relative, color=color, marker=marker, label=task,
                linewidth=8, markersize=20, markeredgewidth=1,
                markeredgecolor=color, solid_capstyle='round',
                solid_joinstyle='round', zorder=3)
    all_sizes = np.concatenate([values[0] for values in groups.values()])
    ax.set_xscale('log')
    ax.set_xlim(all_sizes.min() / 1.15, all_sizes.max() * 1.15)
    ax.set_ylim(-0.5, 21)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel('Pre-training Dataset Size', fontsize=50, labelpad=15)
    ax.set_ylabel('Relative Improvement (%)', fontsize=50, labelpad=12)
    ax.set_title('Scaling Behavior', fontsize=50, pad=15)
    ax.tick_params(which='major', direction='in', labelsize=30, width=2, length=7)
    ax.tick_params(which='minor', direction='in', width=1.5, length=4)
    ax.tick_params(axis='x', which='major', pad=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('#333333')
    ax.legend(fontsize=28, frameon=False, loc='upper right', handlelength=2.2,
              markerscale=1.1, labelspacing=0.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=REPO / 'source_data/Source_Data_Fig2c.csv')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/fig2/Fig2c.png')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    groups = load_data(args.input)
    for task, (_, relative) in groups.items():
        print(f'{task}: relative improvements (%) = {relative.tolist()}')
    fig = plot(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
