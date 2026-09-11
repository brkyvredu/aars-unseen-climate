from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description='Reproduce the physical HDD18–CDD22 climate-space plot.')
    p.add_argument('--source', type=Path, default=ROOT/'data/processed/energy_results_classguard_v2_6000_physics.csv')
    p.add_argument('--external', type=Path, default=ROOT/'data/processed/energy_results_external_8000_physics.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT/'figures/generated')
    return p.parse_args()


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    cols = ['city','climate_heating_degree_days_18c','climate_cooling_degree_days_22c']
    s = pd.read_csv(a.source, usecols=cols).drop_duplicates('city').copy()
    e = pd.read_csv(a.external, usecols=cols).drop_duplicates('city').copy()

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(8.3, 5.6))
    ax.scatter(s[cols[1]], s[cols[2]], marker='s', s=55, label='Source climates')
    ax.scatter(e[cols[1]], e[cols[2]], marker='o', s=55, label='Held-out climates')
    for df in (s, e):
        for _, r in df.iterrows():
            ax.annotate(str(r.city), (r[cols[1]], r[cols[2]]), xytext=(5,5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Heating degree days, HDD18 (°C·day)')
    ax.set_ylabel('Cooling degree days, CDD22 (°C·day)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.output_dir/'fig02_climate_space.pdf')
    fig.savefig(a.output_dir/'fig02_climate_space.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
