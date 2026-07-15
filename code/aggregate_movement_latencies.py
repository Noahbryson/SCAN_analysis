import argparse
import os
from pathlib import Path
from platform import system

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.SCAN_group_analysis import SCAN_group_analysis

plt.rcParams['font.family'] = 'Helvetica'


def parse_args() -> argparse.Namespace:
      parser = argparse.ArgumentParser(
            description="Plot baseline group movement latency distributions across SCAN subjects.",
      )
      parser.add_argument(
            "--outdir",
            type=Path,
            default=None,
            help="Output directory. Defaults to SCAN_Mayo/group_figs/movement_latencies.",
      )
      return parser.parse_args()


def get_boxpath() -> Path:
      userpath = Path(os.path.expanduser('~'))
      if system() == 'Windows':
            return userpath
      return userpath / "Library" / "CloudStorage" / "Box-Box"


def build_subject_latency_summary(latency_data: pd.DataFrame) -> pd.DataFrame:
      summary = (
            latency_data
            .groupby(['subject', 'Movement'], as_index=False)
            .agg(
                  mean_latency=('Latency', 'mean'),
                  median_latency=('Latency', 'median'),
                  n_trials=('Latency', 'count'),
            )
      )
      return summary


def plot_group_latency_distributions(
      latency_data: pd.DataFrame,
      subject_summary: pd.DataFrame,
) -> plt.Figure:
      fig, axes = plt.subplots(1, 3, figsize=(16, 6))
      movement_order = sorted(latency_data['Movement'].dropna().unique().tolist())
      movement_colors = dict(zip(movement_order, sns.color_palette('Set2', n_colors=len(movement_order))))

      sns.violinplot(
            data=latency_data,
            x='Movement',
            y='Latency',
            inner=None,
            cut=0,
            ax=axes[1],
            color=(0.78, 0.84, 0.96),
      )
      sns.boxplot(
            data=latency_data,
            x='Movement',
            y='Latency',
            width=0.5,
            showcaps=True,
            ax=axes[0],
      )
      # sns.boxplot(
      #       data=latency_data,
      #       x='Movement',
      #       y='Latency',
      #       width=0.25,
      #       showcaps=True,
      #       boxprops={'facecolor': 'white', 'zorder': 3},
      #       ax=axes[0],
      # )
      sns.swarmplot(
            data=subject_summary,
            x='Movement',
            y='mean_latency',
            hue='subject',
            dodge=False,
            size=5,
            ax=axes[1],
      )
      sns.swarmplot(
            data=subject_summary,
            x='Movement',
            y='median_latency',
            hue='subject',
            dodge=False,
            size=5,
            ax=axes[0],
      )
      axes[0].set_title('Subject Median Movement Latencies')
      axes[1].set_title('Subject Mean Movement Latencies')
      axes[0].set_ylabel('Latency (ms)')
      axes[1].set_ylabel('Latency (ms)')
      axes[0].set_xlabel('')
      axes[1].set_xlabel('')
      axes[0].legend(title='Subject', bbox_to_anchor=(1.02, 1.0), loc='upper right')
      axes[0].set_ylim([0,2500])
      axes[1].set_ylim([0,2500])

      for movement in movement_order:
            subset = latency_data.loc[latency_data['Movement'] == movement]
            sns.histplot(
                  data=subset,
                  x='Latency',
                  bins=25,
                  stat='density',
                  element='bars',
                  alpha=0.25,
                  color=movement_colors[movement],
                  ax=axes[-1],
                  label=movement,
            )
            sns.kdeplot(
                  data=subset,
                  x='Latency',
                  color=movement_colors[movement],
                  linewidth=2.0,
                  ax=axes[-1],
            )
      axes[-1].set_title('Trial-Level Latency Distribution')
      axes[-1].set_xlabel('Latency (ms)')
      axes[-1].set_ylabel('Density')
      axes[-1].legend(title='Movement')

      for ax in axes:
            ax.spines[['right', 'top']].set_visible(False)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)
            ax.tick_params(axis='both', labelsize=8)

      fig.tight_layout()
      return fig


def main() -> None:
      args = parse_args()
      dataroot = get_boxpath() / 'Brunner Lab' / 'DATA' / 'SCAN_Mayo'
      subjects_file = dataroot / 'subjects.json'
      latency_root = dataroot / 'Aggregate' / 'movement_latencies'
      outdir = args.outdir or dataroot / 'group_figs' / 'movement_latencies'
      outdir.mkdir(parents=True, exist_ok=True)

      group_data = SCAN_group_analysis(latency_root, subjects_file)
      latency_data = group_data.load_latencies().dropna(subset=['Latency']).copy()
      latency_data['Latency'] = latency_data['Latency'].astype(float)
      latency_data['subject'] = latency_data['session'].str.split('_').str[0]

      subject_summary = build_subject_latency_summary(latency_data)
      subject_summary.to_csv(outdir / 'group_movement_latency_subject_means.csv', index=False)
      latency_data.to_csv(outdir / 'group_movement_latency_trials.csv', index=False)

      fig = plot_group_latency_distributions(latency_data, subject_summary)
      fig.savefig(outdir / 'group_movement_latency_distribution.png', dpi=300, bbox_inches='tight')
      fig.savefig(outdir / 'group_movement_latency_distribution.svg', bbox_inches='tight')
      
      print(f'Saved movement latency outputs to {outdir}')
      plt.show()


if __name__ == "__main__":
      main()
