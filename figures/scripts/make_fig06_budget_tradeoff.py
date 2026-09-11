from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description='Reproduce the two data panels used for the simulation-budget trade-off figure.')
    p.add_argument('--input', type=Path, default=ROOT/'data/results/strict/accuracy_budget_tradeoff_strict.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT/'figures/generated')
    return p.parse_args()


def style(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y', alpha=0.15)


def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.input).sort_values('target_simulations')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})

    fig, ax = plt.subplots(figsize=(6.6,4.7))
    ax.plot(d.target_simulations, d.mae, marker='o')
    for _, r in d.iterrows():
        ax.annotate(f'{r.mae:.2f}', (r.target_simulations,r.mae), xytext=(0,8), textcoords='offset points', ha='center')
    ax.set_xticks([0,1,3]); ax.set_xlabel('Target-climate EnergyPlus simulations'); ax.set_ylabel('Mean total-energy MAE (kWh/m²-year)'); style(ax)
    fig.tight_layout(); fig.savefig(a.output_dir/'fig06a_prediction_error.pdf'); fig.savefig(a.output_dir/'fig06a_prediction_error.png',dpi=600); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6,4.7))
    ax.plot(d.target_simulations, 100*d.top_recovery, marker='o')
    for _, r in d.iterrows():
        ax.annotate(f'{100*r.top_recovery:.1f}%', (r.target_simulations,100*r.top_recovery), xytext=(0,8), textcoords='offset points', ha='center')
    ax.set_xticks([0,1,3]); ax.set_xlabel('Target-climate EnergyPlus simulations'); ax.set_ylabel('Top-5% design recovery (%)'); style(ax)
    fig.tight_layout(); fig.savefig(a.output_dir/'fig06b_design_recovery.pdf'); fig.savefig(a.output_dir/'fig06b_design_recovery.png',dpi=600); plt.close(fig)


if __name__ == '__main__':
    main()
