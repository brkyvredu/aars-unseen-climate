from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

CITIES = ['Ardahan','Bartin','Bingol','Canakkale','Hatay','Kars','Konya','Sanliurfa']
LABELS = {'Ardahan':'Ardahan','Bartin':'Bartın','Bingol':'Bingöl','Canakkale':'Çanakkale','Hatay':'Hatay','Kars':'Kars','Konya':'Konya','Sanliurfa':'Şanlıurfa'}
SPEC = [
    ('zero_shot_absolute_aligned_1shot','Zero-shot\n1-shot mask',None),
    ('1_shot_global_bias','Global bias\n1-shot','zero_shot_absolute_aligned_1shot'),
    ('1_shot_raw_relative','Raw relative\n1-shot','zero_shot_absolute_aligned_1shot'),
    ('AARS_v4_1_shot','AARS\n1-shot','zero_shot_absolute_aligned_1shot'),
    ('zero_shot_absolute_aligned_3shot','Zero-shot\n3-shot mask',None),
    ('3_shot_global_bias','Global bias\n3-shot','zero_shot_absolute_aligned_3shot'),
    ('3_shot_raw_relative','Raw relative\n3-shot','zero_shot_absolute_aligned_3shot'),
    ('AARS_v4_3_shot','AARS\n3-shot','zero_shot_absolute_aligned_3shot'),
]


def parse_args():
    p=argparse.ArgumentParser(description='Reproduce the strict total-energy MAE heatmap.')
    p.add_argument('--input',type=Path,default=ROOT/'data/results/strict/external_all_methods_by_city_strict.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'figures/generated')
    return p.parse_args()


def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input); df=df[df.target.eq('total')].copy()
    mae=pd.DataFrame(index=CITIES); pct=pd.DataFrame(index=CITIES)
    for protocol,label,ref_protocol in SPEC:
        vals=df[df.protocol.eq(protocol)].set_index('target_city').loc[CITIES,'mae'].astype(float)
        mae[label]=vals
        if ref_protocol is None:
            pct[label]=np.nan
        else:
            ref=df[df.protocol.eq(ref_protocol)].set_index('target_city').loc[CITIES,'mae'].astype(float)
            pct[label]=100*(ref-vals)/ref

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9.5,'xtick.labelsize':8.2,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(11.2,6.0))
    im=ax.imshow(np.log10(mae.to_numpy(float)),aspect='auto')
    ax.set_xticks(np.arange(len(mae.columns))); ax.set_xticklabels(mae.columns)
    ax.set_yticks(np.arange(len(CITIES))); ax.set_yticklabels([LABELS[c] for c in CITIES])
    ax.axvline(3.5,linewidth=1.2)
    ax.text(1.5,-0.82,'1-shot evaluation universe',ha='center',va='center',fontsize=8.5,fontweight='bold')
    ax.text(5.5,-0.82,'3-shot evaluation universe',ha='center',va='center',fontsize=8.5,fontweight='bold')
    for i,city in enumerate(CITIES):
        one=list(mae.columns[1:4]); three=list(mae.columns[5:8])
        one_min=mae.loc[city,one].min(); three_min=mae.loc[city,three].min()
        for j,label in enumerate(mae.columns):
            val=mae.loc[city,label]; ch=pct.loc[city,label]
            if pd.isna(ch): txt=f'{val:.2f}\n(reference)'; fw='normal'
            else:
                sign='−' if ch>=0 else '+'
                txt=f'{val:.2f}\n({sign}{abs(ch):.1f}%)'
                fw='bold' if ((label in one and abs(val-one_min)<1e-12) or (label in three and abs(val-three_min)<1e-12)) else 'normal'
            ax.text(j,i,txt,ha='center',va='center',fontsize=7.4,fontweight=fw)
    cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label('log$_{10}$(MAE)')
    ax.set_xlabel('Adaptation method'); ax.set_ylabel('Held-out climate')
    ax.text(0,1.025,'Cell: MAE (kWh/m²-year); parentheses: reduction vs zero-shot on the same holdout mask',transform=ax.transAxes,ha='left',va='bottom',fontsize=8.2)
    ax.set_xticks(np.arange(-0.5,len(mae.columns),1),minor=True); ax.set_yticks(np.arange(-0.5,len(CITIES),1),minor=True)
    ax.grid(which='minor',linewidth=0.4,alpha=0.25); ax.tick_params(which='minor',bottom=False,left=False)
    fig.tight_layout(); fig.savefig(a.output_dir/'fig04_total_energy_mae_with_relative_change.pdf'); fig.savefig(a.output_dir/'fig04_total_energy_mae_with_relative_change.png',dpi=600); plt.close(fig)


if __name__=='__main__': main()
