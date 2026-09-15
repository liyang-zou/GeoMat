"""Plot Fig. 2b from the recorded five-fold MAE means and standard deviations."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = ['train_fraction_pct', 'mae_pretrained_mean', 'mae_pretrained_sd',
                   'mae_scratch_mean', 'mae_scratch_sd']
        if not {'task', 'unit', *columns}.issubset(reader.fieldnames or []):
            raise ValueError('Input CSV is missing task, unit, or MAE summary columns')
        rows = list(reader)
    if not rows:
        raise ValueError('Input CSV contains no observations')
    groups = {}
    for task in dict.fromkeys(row['task'] for row in rows):
        values = sorted((row for row in rows if row['task'] == task),
                        key=lambda row: float(row['train_fraction_pct']))
        matrix = np.array([[float(row[col]) for col in columns] for row in values])
        fractions, pm, ps, sm, ss = matrix.T
        if (not np.isfinite(matrix).all() or (fractions <= 0).any()
                or (fractions > 100).any() or len(set(fractions)) != len(fractions)
                or 100 not in fractions or (ps < 0).any() or (ss < 0).any()
                or (pm - ps <= 0).any() or (sm - ss <= 0).any()):
            raise ValueError(f'{task}: invalid fractions or MAE ranges for log axes')
        groups[task] = matrix
    return groups


def plot(groups):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 12})
    fig, axes = plt.subplots(len(groups), 1, sharex=True,
                             figsize=(8, 14), squeeze=False)
    labels = {'JDFT2D': 'MAE (meV/atom)',
              'GVRH': r'MAE ($\log_{10}\mathrm{VRH}$)',
              'MP_E_Form': 'MAE (eV/atom)'}
    legend = [Line2D([0], [0], color='#1f77b4', marker='o', label='Pre-trained'),
              Line2D([0], [0], color='#ff7f0e', marker='s', label='From Scratch'),
              Line2D([0], [0], color='#d62728', linestyle='--', label='Baseline')]
    for index, (task, matrix) in enumerate(groups.items()):
        ax = axes[index, 0]
        fractions, pm, ps, sm, ss = matrix.T
        ax.plot(fractions, pm, marker='o', color='#1f77b4', linewidth=2)
        ax.fill_between(fractions, pm - ps, pm + ps, color='#1f77b4', alpha=0.2)
        ax.plot(fractions, sm, marker='s', color='#ff7f0e', linewidth=2)
        ax.fill_between(fractions, sm - ss, sm + ss, color='#ff7f0e', alpha=0.2)
        baseline = sm[np.flatnonzero(fractions == 100)[0]]
        ax.axhline(baseline, color='#d62728', linestyle='--', linewidth=1.5, zorder=0)
        ax.set_ylabel(labels.get(task, 'MAE'), fontsize=28, labelpad=10)
        ax.text(0.5, 0.96, task, transform=ax.transAxes, ha='center', va='top',
                fontsize=20, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(4.7, 103)
        ax.set_xticks(fractions)
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(which='major', direction='in', labelsize=20, width=2, length=6)
        ax.tick_params(which='minor', direction='in', labelsize=20, width=1.5, length=3)
        if index == 0:
            ax.legend(handles=legend, fontsize=16, frameon=False, loc='upper right')
    axes[-1, 0].set_xlabel('Fraction of Training Data (%)', fontsize=28, labelpad=10)
    fig.tight_layout(h_pad=0.15)
    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.2, 0.5)
    fig.subplots_adjust(left=0.21, hspace=0.04)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=REPO / 'source_data/Source_Data_Fig2b.csv')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/fig2/Fig2b.png')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    groups = load_data(args.input)
    fig = plot(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
