from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description='Reproduce the aligned zero-shot to 3-shot AARS city-level recovery plot.')
    p.add_argument('--input', type=Path, default=ROOT/'data/results/strict/external_all_methods_by_city_strict.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT/'figures/generated')
    return p.parse_args()


def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.input)
    d = d[d.target.eq('total')]
    z = d[d.protocol.eq('zero_shot_absolute_aligned_3shot')][['target_city','mae']].rename(columns={'mae':'zero'})
    m = d[d.protocol.eq('AARS_v4_3_shot')][['target_city','mae']].rename(columns={'mae':'aars'})
    x = z.merge(m, on='target_city')
    x['reduction_pct'] = 100*(x.zero-x.aars)/x.zero
    x = x.sort_values('zero').reset_index(drop=True)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(x))
    for i, r in x.iterrows():
        ax.plot([r.aars, r.zero], [i,i], linewidth=0.9, alpha=0.45)
        ax.text((r.aars+r.zero)/2, i+0.12, f'−{r.reduction_pct:.0f}%', ha='center', fontsize=7.5)
    ax.scatter(x.zero, y, marker='o', s=50, label='Aligned zero-shot')
    ax.scatter(x.aars, y, marker='D', s=50, label='AARS 3-shot')
    ax.set_yticks(y); ax.set_yticklabels(x.target_city)
    ax.set_xlabel('Total-energy MAE (kWh/m²-year)')
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.15)
    ax.legend(frameon=False)
    ax.text(0.99, 0.02, 'Common 997-design holdout universe', transform=ax.transAxes, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(a.output_dir/'fig03_zero_shot_recovery.pdf')
    fig.savefig(a.output_dir/'fig03_zero_shot_recovery.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
