"""Plot Fig. 4d-e from precomputed t-SNE coordinates."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
PANELS = {
    'd': {
        'source': 'Source_Data_Fig4d.csv',
        'task': 'MP_E_Form',
    },
    'e': {
        'source': 'Source_Data_Fig4e.csv',
        'task': 'MP_Is_Metal',
    },
}


def load_source_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'tsne_1', 'tsne_2', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Input CSV must contain {sorted(required)}')
        rows = list(reader)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input CSV is empty or contains duplicate IDs')
    values = np.array([
        [float(row['tsne_1']), float(row['tsne_2']), float(row['label'])]
        for row in rows
    ])
    if not np.isfinite(values).all():
        raise ValueError('t-SNE coordinates and labels must be finite')
    return values


def draw_panel(ax, panel, values):
    coordinates, labels = values[:, :2], values[:, 2]
    if panel == 'd':
        scatter = ax.scatter(
            coordinates[:, 0], coordinates[:, 1], c=labels,
            cmap='coolwarm', s=10, alpha=0.7, linewidths=0,
            rasterized=True)
        colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label('Formation Energy (eV/atom)', fontsize=16)
        colorbar.ax.tick_params(labelsize=12)
    else:
        unique_labels = set(np.unique(labels))
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError('Fig. 4e labels must be binary (0 or 1)')
        cmap = plt.get_cmap('coolwarm')
        for label, name, color in (
                (0.0, 'Nonmetal', cmap(0.0)),
                (1.0, 'Metal', cmap(1.0))):
            mask = labels == label
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1],
                       color=[color], label=name, s=10, alpha=0.7,
                       linewidths=0, rasterized=True)
        ax.legend(fontsize=16, loc='best', markerscale=2)
    ax.set_title(PANELS[panel]['task'], fontsize=20, pad=8)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=('d', 'e', 'all'), default='all')
    parser.add_argument('--input', type=Path,
                        help='Override Source Data for one selected panel')
    parser.add_argument('--output', type=Path,
                        help='Override the default output image')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.input is not None and args.panel == 'all':
        parser.error('--input can only be used with --panel d or e')

    selected = tuple(PANELS) if args.panel == 'all' else (args.panel,)
    fig, axes = plt.subplots(
        1, len(selected), figsize=(9 * len(selected), 8), squeeze=False)
    for ax, panel in zip(axes[0], selected):
        source = (args.input if args.input is not None else
                  REPO / 'source_data' / PANELS[panel]['source'])
        values = load_source_data(source)
        draw_panel(ax, panel, values)
        print(f'Fig. 4{panel}: {len(values)} points from {source}')
    fig.tight_layout()

    default_name = 'Fig4d_e.png' if args.panel == 'all' else f'Fig4{args.panel}.png'
    output = args.output or REPO / 'outputs' / 'fig4' / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
