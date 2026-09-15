"""Plot Fig. 2a from the recorded scratch and pretrained MAE values."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'task', 'backbone', 'unit', 'mae_scratch', 'mae_pretrained'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Input CSV must contain {sorted(required)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Input CSV contains no observations')
    tasks = list(dict.fromkeys(row['task'] for row in rows))
    backbones = list(dict.fromkeys(row['backbone'] for row in rows))
    lookup = {(row['backbone'], row['task']): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate backbone/task observations')
    gains = {}
    for backbone in backbones:
        pairs = [lookup[(backbone, task)] for task in tasks]
        scratch = np.array([float(row['mae_scratch']) for row in pairs])
        pretrained = np.array([float(row['mae_pretrained']) for row in pairs])
        if (not np.isfinite(scratch).all() or not np.isfinite(pretrained).all()
                or (scratch <= 0).any() or (pretrained < 0).any()):
            raise ValueError('MAE values must be finite, with positive scratch MAE')
        gains[backbone] = (scratch - pretrained) / scratch * 100
    return tasks, gains


def plot(tasks, gains):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 12})
    theta = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    closed_theta = np.append(theta, theta[0])
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.plot(closed_theta, np.zeros(len(tasks) + 1), color='k',
            linestyle='--', linewidth=1, alpha=0.5)
    colors = {'CGCNN': '#d62728', 'MEGNET': '#2ca02c', 'PaiNN': '#1f77b4'}
    for backbone, values in gains.items():
        color = colors.get(backbone)
        closed_values = np.append(values, values[0])
        ax.plot(closed_theta, closed_values, color=color, linewidth=2, label=backbone)
        ax.fill(closed_theta, closed_values, color=color,
                alpha=0.25 if backbone == 'PaiNN' else 0.15)
    all_values = np.concatenate(list(gains.values()))
    ax.set_ylim(min(0, float(all_values.min())), float(all_values.max()) + 1)
    ax.set_rgrids([10, 20, 30, 40], ['10%', '20%', '30%', '40%'],
                 angle=0, fontsize=20, color='black')
    ax.set_thetagrids(np.degrees(theta), tasks, fontsize=22, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=20)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=REPO / 'source_data/Source_Data_Fig2a.csv')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/fig2/Fig2a.png')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    tasks, gains = load_data(args.input)
    for backbone, values in gains.items():
        print(f'{backbone}: mean relative improvement = {values.mean():.6f}%')
    fig = plot(tasks, gains)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
