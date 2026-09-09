import glob
import os
import platform
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PyBrain.modules.surface_projection import Atlas, projectAtlas
from src.SCAN_SingleSessionAnalysis import *
from src.functions.graphics import circle_gradient_key, default_gradient
from src.modules.colorOps import colorOps


localEnv = platform.system()
userPath = Path(os.path.expanduser("~"))
if localEnv == "Windows":
    boxPath = userPath / "Box"
else:
    boxPath = userPath / "Library/CloudStorage/Box-Box"
dataPath = boxPath / "Brunner Lab/DATA/SCAN_Mayo"

subject = "BJH041"
gammaRange = [70, 170]
session = "pre_ablation"
atlas_res = "32k"
SCAN_ROI = (
    boxPath
    / "Brunner Lab/patients/BJH041/Freesurfer&Workbench/MSC18_SCANROIs.dtseries.nii"
)
lookup_keys = ["interop_RFA_FLAIR", "LITT"]
og_ablation_keys = ["RF_Volume", "RF_edema", "RF_M12-14", "RF_L14-16"]
RF_color = mcolors.to_rgb(r"#FFDA03")
flatmap_xlim = [-80, 80]
flatmap_ylim = [-105, 170]
point_size = 120.0

laplacian = False
bipolar = True
loadData = True
save = False
save_figures = True

if bipolar:
    reref = "bipolar"
elif laplacian:
    reref = "laplacian"
else:
    reref = "common"


def build_subject_flatmap(subject_id: str) -> tuple[projectAtlas, str | None, list[str]]:
    atlas: Atlas = Atlas.fs_LR_from_fsav(atlas_res)
    atlas.pt_sphere_name = "sphere.reg.surf.gii"
    seg = f"/Users/nkb/Documents/NCAN/patients/{subject_id}/segmentation"
    electrodes_dir = f"/Users/nkb/Documents/NCAN/patients/{subject_id}/electrodes_clean"
    if not os.path.exists(electrodes_dir):
        electrodes_dir = f"/Users/nkb/Documents/NCAN/patients/{subject_id}/electrodes"
    label_path = Path(f"/Users/nkb/Documents/NCAN/patients/{subject_id}/imaging/labels")
    pattern = glob.glob(str(label_path / "*RF*")) + glob.glob(str(label_path / "*LITT*"))
    filetree = boxPath / subject_id

    flatmap = projectAtlas(
        seg,
        atlas=atlas,
        electrode_dir=electrodes_dir,
        process_fs=False,
        process_gifti=False,
        buildFileTree=filetree,
        distanceThreshold=5,
        correct_affine=False,
    )
    lesion_paths = flatmap.project_freesurfer_labels(label_files=pattern, process=False)
    target_key = None
    if lesion_paths:
        target_matches = [key for key in lesion_paths for tag in lookup_keys if tag in key]
        if target_matches:
            target_key = target_matches[0]
            for hemi_paths in lesion_paths[target_key].values():
                flatmap.load_binary_gifti_as_ROI(hemi_paths["fs_LR"], target_key)
            flatmap.update_ROI_cmap(target_key, {target_key: RF_color})
    other_lesion_keys = [
        key
        for key in lesion_paths
        if key != target_key and any(tag in key for tag in og_ablation_keys)
    ]
    for lesion_key in other_lesion_keys:
        for hemi_paths in lesion_paths[lesion_key].values():
            flatmap.load_binary_gifti_as_ROI(hemi_paths["fs_LR"], lesion_key)
        flatmap.update_ROI_cmap(lesion_key, {lesion_key: (0.6, 0.6, 0.6)})

    flatmap.load_cifti_as_ROI(str(SCAN_ROI), "SCAN")
    flatmap.update_ROI_cmap("SCAN", {"SCAN": (1, 1, 1)})
    return flatmap, target_key, other_lesion_keys


def make_effect_plot_df(
    flatmap: projectAtlas,
    effect_df: pd.DataFrame,
    color_lookup: dict[str, tuple[float, float, float] | tuple[float, float, float, float]],
    alpha_lookup: dict[str, float],
) -> pd.DataFrame:
    electrode_df = flatmap.electrode_library.copy()
    if "names" not in electrode_df.columns:
        electrode_df = electrode_df.reset_index().rename(columns={"index": "names"})
    else:
        electrode_df = electrode_df.reset_index(drop=True)
    merged = effect_df.merge(electrode_df, on="names", how="inner")
    merged["color"] = merged["names"].map(color_lookup)
    merged["alpha"] = merged["names"].map(alpha_lookup)
    return merged.dropna(subset=["color", "alpha"])


def scale_metric_to_alpha(
    effect_df: pd.DataFrame,
    metric_column: str,
    significant_column: str = "Significant",
    nonsig_alpha: float = 0.1,
    min_sig_alpha: float = 0.4,
    max_sig_alpha: float = 0.95,
) -> dict[str, float]:
    metric = effect_df[metric_column].astype(float)
    metric_min = float(metric.min())
    metric_max = float(metric.max())
    if np.isclose(metric_min, metric_max):
        scaled = np.ones(len(metric))
    else:
        scaled = (metric - metric_min) / (metric_max - metric_min)
    out: dict[str, float] = {}
    for row, frac in zip(effect_df.itertuples(index=False), scaled):
        is_sig = bool(getattr(row, significant_column))
        if is_sig:
            out[getattr(row, "names")] = float(min_sig_alpha + frac * (max_sig_alpha - min_sig_alpha))
        else:
            out[getattr(row, "names")] = float(nonsig_alpha)
    return out


def constant_alpha_lookup(names: list[str], alpha: float) -> dict[str, float]:
    return {name: alpha for name in names}


def metric_to_color_lookup(
    effect_df: pd.DataFrame,
    metric_column: str,
    cmap: mcolors.Colormap,
) -> dict[str, tuple[float, float, float, float]]:
    metric = effect_df[metric_column].astype(float)
    metric_min = float(metric.min())
    metric_max = float(metric.max())
    if np.isclose(metric_min, metric_max):
        scaled = np.ones(len(metric))
    else:
        scaled = (metric - metric_min) / (metric_max - metric_min)
    return {
        name: cmap(float(frac))
        for name, frac in zip(effect_df["names"].to_list(), scaled)
    }


def plot_effect_flatmaps(
    flatmap: projectAtlas,
    effect_df: pd.DataFrame,
    outdir: Path,
    stem: str,
    title: str,
    color_lookup: dict[str, tuple[float, float, float] | tuple[float, float, float, float]],
    alpha_lookup: dict[str, float],
    target_key: str | None = None,
    other_lesion_keys: list[str] | None = None,
) -> None:
    plot_df = make_effect_plot_df(flatmap, effect_df, color_lookup, alpha_lookup)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    side_map = [("L", "lh"), ("R", "rh")]

    for ax, (side, hemi) in zip(axes, side_map):
        flatmap.flatmap_plot(side, ["S_central", "G_precentral"], legend=False, ax=ax)
        flatmap.flatplot_additional_ROI("SCAN", side, ax=ax, showLegend=False, opacity=1.0)
        if target_key is not None:
            flatmap.flatplot_additional_ROI(target_key, side, ax=ax, showLegend=False, opacity=0.35)
        if other_lesion_keys is not None:
            for lesion_key in other_lesion_keys:
                flatmap.flatplot_additional_ROI(
                    lesion_key, side, ax=ax, showLegend=False, opacity=0.25, outline=True
                )

        hemi_df = plot_df.loc[plot_df["hemi"] == hemi].copy()
        if not hemi_df.empty:
            flatmap.plot_electrodes(
                ax=ax,
                hemi=hemi,
                bipolar=bipolar,
                subset=["names", hemi_df["names"].tolist()],
                color=hemi_df["color"].to_list(),
                size=point_size,
                alphas=hemi_df["alpha"].to_list(),
            )
            flatmap.fit_image_to_electrode(ax, hemi_df)
        ax.set_xlim(flatmap_xlim)
        ax.set_ylim(flatmap_ylim)
        ax.set_title(f"{title} {side}")

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.9, wspace=0.08)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


bp = dataPath / subject / "brain" / "brain_cortex.mat"
flatmap, target_key, other_lesion_keys = build_subject_flatmap(subject)

a = SCAN_SingleSessionAnalysis(
    dataPath,
    subject,
    session,
    remove_trajectories=["OR"],
    load=loadData,
    plot_stimuli=False,
    gammaRange=gammaRange,
    refType=reref,
)
r_sq, p_vals, U_res, d_res, roc_res = a.task_power_analysis(save=save)
sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals, alpha=0.05)
cmap_resolution = 1
tuning_colors, target_colors = default_gradient(cmap_resolution)
tuning, chan_tuning_colors, angle_key, color_array = a.somatotopic_tuning(
    r_sq, tuning_colors=tuning_colors, plotCMAP=True
)

colorLab = list(angle_key.keys())
labColor = [i["color"] for i in angle_key.values()]
t_ = [[i, j["color"]] for i, j in angle_key.items()]
t_names = [i[0] for i in t_]
t_colors = [i[1] for i in t_]

effect_of_interest = r_sq
datasubset = sig_chans
shared_rep = a.shared_representation(effect_of_interest, sig_chans)
intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(
    effect_of_interest, datasubset, save=save, label="significant", thresh=0.1, comparison=""
)
intereffectors = [i.replace("_", "") for i in intereffectors]
nonspecifics = [i.replace("_", "") for i in nonspecifics]

print("intereffectors")
for name in intereffectors:
    print(name)
print("','".join(intereffectors))

print("\n\nnonspecifics")
non_specific = ["inter", "foot-face", "hand-face", "hand-foot"]
multiMotor = [[i, j] for i, j in channel_classifcation.items() if j in non_specific]
targets = []
for name, label in multiMotor:
    print(name, label)
    targets.append(name)

allChans = tuning["channel"].to_list()
tuning_plot_df = tuning.rename(columns={"channel": "names"}).merge(
    shared_rep.rename(columns={"channel": "names"})[["names", "Shared Rep", "Significant"]],
    on="names",
    how="left",
)
shared_rep_plot_df = shared_rep.rename(columns={"channel": "names"})

if save_figures:
    figure_root = (
        boxPath
        / "Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/2.Multi-modal Mapping/temp/flatmaps"
    )
    shared_rep_path = figure_root / "shared_rep"
    tuning_path = figure_root / "tuning"
    RMA_path = figure_root / "RMA"
    for outdir in [shared_rep_path, tuning_path, RMA_path]:
        outdir.mkdir(parents=True, exist_ok=True)

    tcolor = colorOps().cmyk2rgb([4.17, 23.72, 0.06, 0])

    shared_rep_colors = metric_to_color_lookup(shared_rep_plot_df, "Shared Rep", plt.cm.viridis)
    shared_rep_alpha = scale_metric_to_alpha(shared_rep_plot_df, "Shared Rep")
    plot_effect_flatmaps(
        flatmap=flatmap,
        effect_df=shared_rep_plot_df,
        outdir=shared_rep_path,
        stem="shared_representation_flatmap",
        title="Shared Representation",
        color_lookup=shared_rep_colors,
        alpha_lookup=shared_rep_alpha,
        target_key=target_key,
        other_lesion_keys=other_lesion_keys,
    )

    fig = circle_gradient_key(tuning_colors, target_names=t_names, target_colors=t_colors)
    fig.savefig(tuning_path / "colorwheel.svg", format="svg")
    plt.close(fig)

    tuning_colors_lookup = {row.names: chan_tuning_colors[row.names] for row in tuning_plot_df.itertuples(index=False)}
    tuning_alpha = scale_metric_to_alpha(tuning_plot_df, "Shared Rep")
    plot_effect_flatmaps(
        flatmap=flatmap,
        effect_df=tuning_plot_df,
        outdir=tuning_path,
        stem="somatotopic_tuning_flatmap",
        title="Somatotopic Tuning",
        color_lookup=tuning_colors_lookup,
        alpha_lookup=tuning_alpha,
        target_key=target_key,
        other_lesion_keys=other_lesion_keys,
    )

    rma_target_df = pd.DataFrame({"names": intereffectors})
    rma_color_lookup = {name: tcolor for name in intereffectors}
    plot_effect_flatmaps(
        flatmap=flatmap,
        effect_df=rma_target_df,
        outdir=RMA_path,
        stem="RMA_locs_flatmap",
        title="RMA Locations",
        color_lookup=rma_color_lookup,
        alpha_lookup=constant_alpha_lookup(intereffectors, 0.95),
        target_key=target_key,
        other_lesion_keys=other_lesion_keys,
    )

    target_rma = [name for name in intereffectors if "KL" in name]
    target_rma_df = pd.DataFrame({"names": target_rma})
    target_rma_colors = {name: chan_tuning_colors[name] for name in target_rma if name in chan_tuning_colors}
    plot_effect_flatmaps(
        flatmap=flatmap,
        effect_df=target_rma_df,
        outdir=RMA_path,
        stem="target_RMA_tuning_flatmap",
        title="Target RMA Tuning",
        color_lookup=target_rma_colors,
        alpha_lookup=constant_alpha_lookup(target_rma, 0.95),
        target_key=target_key,
        other_lesion_keys=other_lesion_keys,
    )

    all_rma_df = pd.DataFrame({"names": intereffectors})
    all_rma_colors = {name: chan_tuning_colors[name] for name in intereffectors if name in chan_tuning_colors}
    plot_effect_flatmaps(
        flatmap=flatmap,
        effect_df=all_rma_df,
        outdir=RMA_path,
        stem="RMA_tuning_flatmap",
        title="RMA Tuning",
        color_lookup=all_rma_colors,
        alpha_lookup=constant_alpha_lookup(intereffectors, 0.95),
        target_key=target_key,
        other_lesion_keys=other_lesion_keys,
    )

print(0)
