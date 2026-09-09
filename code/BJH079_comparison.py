from pathlib import Path
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns
import platform

from src.SCAN_group_analysis import SCAN_group_analysis, shared_rep, tuning


def save_fig(fig: plt.Figure, outdir: Path, name: str) -> None:
      fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
      fig.savefig(outdir / f"{name}.svg", bbox_inches="tight")


def prepare_metric_data(group_analysis: SCAN_group_analysis, reference_session: str) -> pd.DataFrame:
      data = group_analysis.load_rsq(reference_session).copy()
      labels, _ = group_analysis.load_channnel_labels(reference_session)
      data = data.merge(labels, on="channel", suffixes=("", "_label"))
      data["shared_rep"] = data.apply(lambda row: shared_rep(row["Hand"], row["Tongue"], row["Foot"]), axis=1)
      tuning_data = data.apply(
            lambda row: tuning(row["Hand"], row["Foot"], row["Tongue"]),
            axis=1,
            result_type="expand",
      )
      tuning_data.columns = ["tuning", "Magnitude", "Angle"]
      data[["tuning", "Magnitude", "Angle"]] = tuning_data
      data["Magnitude"] = data["Magnitude"].astype(float)
      data["Angle"] = data["Angle"].astype(float)
      data["class_group"] = data["class"].apply(lambda value: "non-specific" if "-" in value else value)
      return data


def build_paired_metric(metric_data: pd.DataFrame, session_a: str, session_b: str, metric: str, significant: bool = True) -> pd.DataFrame:
      data = metric_data.copy()
      if significant:
            data = data.loc[data["significant"] == True].copy()
      a = data.loc[data["session"] == session_a, ["channel", "class_group", metric]].copy()
      b = data.loc[data["session"] == session_b, ["channel", "class_group", metric]].copy()
      paired = a.merge(b, on=["channel", "class_group"], suffixes=("_a", "_b"))
      paired = paired.dropna(subset=[f"{metric}_a", f"{metric}_b"])
      return paired


def run_paired_test(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
      if len(values_a) == 0 or len(values_b) == 0:
            return np.nan, np.nan
      if len(values_a) != len(values_b):
            return np.nan, np.nan
      if np.allclose(values_a, values_b, equal_nan=True):
            return 0.0, 1.0
      res = stats.wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
      return float(res.statistic), float(res.pvalue)


def plot_paired_metric(
      metric_data: pd.DataFrame,
      session_a: str,
      session_b: str,
      metric: str,
      title: str,
      ylabel: str,
      ylim: tuple[float, float],
      significant: bool = True,
) -> plt.Figure:
      paired = build_paired_metric(metric_data, session_a, session_b, metric, significant=significant)
      class_order = [label for label in ["hand", "foot", "tongue", "non-specific"] if label in paired["class_group"].unique()]
      if len(class_order) == 0:
            fig, ax = plt.subplots(1, 1)
            ax.text(0.5, 0.5, f"No paired data for {metric}", ha="center", va="center")
            ax.set_axis_off()
            return fig

      fig, axs = plt.subplots(1, len(class_order), sharex=False, sharey=True, figsize=(4 * len(class_order), 5))
      if len(class_order) == 1:
            axs = np.array([axs])

      colors = sns.color_palette("Set2", n_colors=2)
      for ax, class_name in zip(axs, class_order):
            df = paired.loc[paired["class_group"] == class_name].copy()
            values_a = df[f"{metric}_a"].to_numpy(dtype=float)
            values_b = df[f"{metric}_b"].to_numpy(dtype=float)
            _, pvalue = run_paired_test(values_a, values_b)

            for _, row in df.iterrows():
                  ax.plot([0, 1], [row[f"{metric}_a"], row[f"{metric}_b"]], color=(0.7, 0.7, 0.7), alpha=0.7, linewidth=1)
            ax.scatter(np.zeros(len(df)), values_a, color=colors[0], s=35)
            ax.scatter(np.ones(len(df)), values_b, color=colors[1], s=35)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([session_a, session_b], rotation=45)
            ax.set_title(f"{class_name}\nN={len(df)}\np={pvalue:.4g}" if not np.isnan(pvalue) else f"{class_name}\nN={len(df)}")
            ax.set_ylim(ylim)
            ax.spines[["right", "top"]].set_visible(False)

      axs[0].set_ylabel(ylabel)
      fig.suptitle(title)
      fig.tight_layout()
      return fig


def plot_rsq(metric_data: pd.DataFrame, session_a: str, session_b: str) -> plt.Figure:
      movements = ["Hand", "Foot", "Tongue"]
      paired_frames = []
      for movement in movements:
            df = build_paired_metric(metric_data, session_a, session_b, movement, significant=True)
            df["movement"] = movement
            paired_frames.append(df)
      paired = pd.concat(paired_frames, ignore_index=True)
      class_order = [label for label in ["hand", "foot", "tongue", "non-specific"] if label in paired["class_group"].unique()]

      if len(class_order) == 0:
            fig, ax = plt.subplots(1, 1)
            ax.text(0.5, 0.5, "No paired r^2 data available", ha="center", va="center")
            ax.set_axis_off()
            return fig

      fig, axs = plt.subplots(3, len(class_order), sharex=False, sharey=True, figsize=(4 * len(class_order), 10))
      if len(class_order) == 1:
            axs = np.array(axs).reshape(3, 1)

      colors = sns.color_palette("Set2", n_colors=2)
      for row_idx, movement in enumerate(movements):
            for col_idx, class_name in enumerate(class_order):
                  ax = axs[row_idx, col_idx]
                  df = paired.loc[(paired["movement"] == movement) & (paired["class_group"] == class_name)].copy()
                  values_a = df[f"{movement}_a"].to_numpy(dtype=float)
                  values_b = df[f"{movement}_b"].to_numpy(dtype=float)
                  _, pvalue = run_paired_test(values_a, values_b)

                  for _, row in df.iterrows():
                        ax.plot([0, 1], [row[f"{movement}_a"], row[f"{movement}_b"]], color=(0.7, 0.7, 0.7), alpha=0.7, linewidth=1)
                  ax.scatter(np.zeros(len(df)), values_a, color=colors[0], s=30)
                  ax.scatter(np.ones(len(df)), values_b, color=colors[1], s=30)
                  ax.set_xticks([0, 1])
                  ax.set_xticklabels([session_a, session_b], rotation=45)
                  ax.set_ylim([-0.5, 1.0])
                  ax.spines[["right", "top"]].set_visible(False)

                  if row_idx == 0:
                        ax.set_title(f"{class_name}\np={pvalue:.4g}")
                  else:
                        ax.set_title(f"p={pvalue:.4g}")
                  if col_idx == 0:
                        ax.set_ylabel(f"{movement} r^2")

      fig.suptitle("Paired r^2 comparison")
      fig.tight_layout()
      return fig


subjectSessions = ["BJH079_aggregate", "BJH079_postRF_aggregate"]
subject = "BJH079"
method = "cluster"

localEnv = platform.system()
userPath = Path(os.path.expanduser("~"))
if localEnv == "Windows":
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
      dataPath = userPath / "Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"

pairPath = dataPath / "Aggregate" / "pairs" / "BJH079_pre-v-post"
erpPath = pairPath / method
os.makedirs(pairPath, exist_ok=True)
os.makedirs(erpPath, exist_ok=True)

a = SCAN_group_analysis(dataPath / "Aggregate", subject_list=subjectSessions)

latencyFig = a.analyze_latencies_pre_post(ylims=[0,4000],drop_movements=['Tongue'])
latencyData = a.load_latencies().dropna(subset=["Latency"]).copy()
latencySummary = (
      latencyData.groupby(["session", "Movement"], as_index=False)
      .agg(
            mean_latency=("Latency", "mean"),
            median_latency=("Latency", "median"),
            n_trials=("Latency", "count"),
      )
)
latencyData.to_csv(pairPath / "latencies_trials.csv", index=False)
latencySummary.to_csv(pairPath / "latencies_summary.csv", index=False)
save_fig(latencyFig, pairPath, "latencies")

# ERP_compare = a.compare_ERP_pair(subjectSessions[0], subjectSessions[1], method=method, savePath=erpPath)

metricData = prepare_metric_data(a, subjectSessions[0])
metricData.to_csv(pairPath / "paired_metrics_long.csv", index=False)

sharedRepData = a.compare_shared_rep(subjectSessions[0], subjectSessions[1], paired=True, significant=False)
sharedRepFig = plt.gcf()
save_fig(sharedRepFig, pairPath, "shared_representation")
sharedRepData.to_csv(pairPath / "shared_representation_long.csv", index=False)

rsqFig = plot_rsq(metricData, subjectSessions[0], subjectSessions[1])
save_fig(rsqFig, pairPath, "r_squared")

tuningMagFig = plot_paired_metric(
      metricData,
      subjectSessions[0],
      subjectSessions[1],
      "Magnitude",
      "Paired somatotopic tuning magnitude",
      "Somatotopic tuning magnitude",
      (-0.5, 1.0),
)
save_fig(tuningMagFig, pairPath, "tuning_magnitude")

tuningAngleFig = plot_paired_metric(
      metricData,
      subjectSessions[0],
      subjectSessions[1],
      "Angle",
      "Paired somatotopic tuning angle",
      "Somatotopic tuning angle (deg)",
      (0.0, 360.0),
)
save_fig(tuningAngleFig, pairPath, "tuning_angle")

plt.show()
