"""Plot the five-fold ablation summaries shown in Fig. 4a-c."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
MODEL_ORDER = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5')
PANELS = {
    'a': {
        'source': 'Source_Data_Fig4a.csv',
        'title': 'Exfoliation Energy (JDFT2D)',
        'ylabel': 'MAE (meV/atom)',
    },
    'b': {
        'source': 'Source_Data_Fig4b.csv',
        'title': 'Shear Modulus (GVRH)',
        'ylabel': r'MAE ($\log_{10}G_{VRH}$)',
    },
    'c': {
        'source': 'Source_Data_Fig4c.csv',
        'title': 'Band Gap (MP_Gap)',
        'ylabel': 'MAE (eV)',
    },
}


def load_summary(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'model', 'mean_mae', 'std_mae'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Input CSV must contain {sorted(required)}')
        rows = list(reader)
    models = tuple(row['model'] for row in rows)
    if models != MODEL_ORDER:
        raise ValueError(f'Models must appear once in this order: {MODEL_ORDER}')
    means = np.array([float(row['mean_mae']) for row in rows])
    standard_deviations = np.array([float(row['std_mae']) for row in rows])
    if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
        raise ValueError('MAE summaries must be finite')
    if (means < 0).any() or (standard_deviations < 0).any():
        raise ValueError('MAE means and standard deviations must be non-negative')
    return means, standard_deviations


def draw_panel(ax, panel, means, standard_deviations):
    config = PANELS[panel]
    colors = ['#aec7e8'] * 5 + ['#1f77b4']
    ax.bar(MODEL_ORDER, means, yerr=standard_deviations, capsize=5,
           color=colors, edgecolor='gray', linewidth=0.8, alpha=0.9,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4})
    ax.set_ylabel(config['ylabel'], fontsize=18)
    ax.set_title(config['title'], fontsize=18, pad=13)
    ax.tick_params(axis='y', which='major', direction='in',
                   labelsize=14, width=1.5, length=4)
    ax.tick_params(axis='x', which='major', direction='in',
                   labelsize=16, width=1.5, length=0)
    ax.set_ylim(bottom=0)


def plot_panels(selected_panels, input_override=None):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })
    width = 5.5 * len(selected_panels)
    fig, axes = plt.subplots(1, len(selected_panels), figsize=(width, 5.3),
                             squeeze=False)
    summaries = {}
    for ax, panel in zip(axes[0], selected_panels):
        source = (input_override if input_override is not None else
                  REPO / 'source_data' / PANELS[panel]['source'])
        means, standard_deviations = load_summary(source)
        draw_panel(ax, panel, means, standard_deviations)
        summaries[panel] = (means, standard_deviations)
    fig.tight_layout()
    return fig, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=('a', 'b', 'c', 'all'), default='all')
    parser.add_argument('--input', type=Path,
                        help='Override the Source Data CSV for one selected panel')
    parser.add_argument('--output', type=Path,
                        help='Override the default output image path')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.input is not None and args.panel == 'all':
        parser.error('--input can only be used with --panel a, b, or c')

    selected_panels = tuple(PANELS) if args.panel == 'all' else (args.panel,)
    default_name = 'Fig4a_c.png' if args.panel == 'all' else f'Fig4{args.panel}.png'
    output_path = args.output or REPO / 'outputs' / 'fig4' / default_name
    fig, summaries = plot_panels(selected_panels, args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    for panel, (means, standard_deviations) in summaries.items():
        values = ', '.join(
            f'{model}={mean:g}+/-{std:g}'
            for model, mean, std in zip(MODEL_ORDER, means, standard_deviations))
        print(f'Fig. 4{panel}: {values}')
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
