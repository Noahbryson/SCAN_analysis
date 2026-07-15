import argparse
import json
import os
import sys
from pathlib import Path
from platform import system
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.axes import Axes
from matplotlib.figure import Figure


REPO_ROOT = Path(__file__).resolve().parents[2]
PYBRAIN_ROOT = REPO_ROOT / "PyBrain"
if str(PYBRAIN_ROOT) not in sys.path:
    sys.path.append(str(PYBRAIN_ROOT))

from PyBrain.modules.surface_projection import Atlas, groupAtlas  # noqa: E402


SCAN_VMAP = {0: "na", 1.5: "inter", 10: "hand", 11: "face", 17: "foot"}
SCAN_CMAP = {
    "na": (0, 0, 0, 0),
    "inter": (158 / 255, 38 / 255, 108 / 255, 1),
    "hand": (68 / 255, 1, 1, 1),
    "face": (1, 142 / 255, 52 / 255, 1),
    "foot": (32 / 255, 133 / 255, 44 / 255, 1),
}
PlotConfigEntry = tuple[str, int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all SCAN patient electrodes on flatmaps and Conte69 very-inflated surfaces.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to SCAN_Mayo/group_figs/SCAN_all_electrodes.",
    )
    parser.add_argument(
        "--atlas-res",
        type=str,
        default="32k",
        help="fs_LR atlas resolution to load.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Surface point size for electrodes.",
    )
    parser.add_argument(
        "--flat-size",
        type=float,
        default=18.0,
        help="Flatmap point size for electrodes.",
    )
    parser.add_argument(
        "--bipolar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bipolar projected electrodes.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Open the PyVista window instead of rendering off-screen only.",
    )
    return parser.parse_args()


def get_boxpath() -> Path:
    userpath = Path(os.path.expanduser("~"))
    if system() == "Windows":
        return userpath
    return userpath / "Library" / "CloudStorage" / "Box-Box"


def get_dataroot() -> Path:
    return get_boxpath() / "Brunner Lab" / "DATA" / "SCAN_Mayo"


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def load_scan_subjects(subjects_file: Path) -> list[str]:
    with open(subjects_file, "r") as fp:
        subjects_info = json.load(fp)
    subjects = subjects_info.get("subjects", [])
    return dedupe_preserve_order(subjects)


def get_conte_surfaces() -> dict[str, Path]:
    conte_tag = "very_inflated"
    conte_root = (
        Path("/Users/nkb/Documents/NCAN/atlases/surface_atlases/CONTE69/32k/surfaces")
    )
    conte_l = (
        conte_root
        / "fs_LR.32k.L_hemi"
        / f"Conte69.L.{conte_tag}.32k_fs_LR.surf.gii"
    )
    conte_r = (
        conte_root
        / "fs_LR.32k.R_hemi"
        / f"Conte69.R.{conte_tag}.32k_fs_LR.surf.gii"
    )
    return {"lh": conte_l, "rh": conte_r}


def get_scan_overlay_path() -> Path:
    return get_dataroot() / "imaging" / "HCP_Spots_Effectors_CS.dtseries.nii"


def get_template_segmentation_dir() -> Path:
    userpath = Path(os.path.expanduser("~"))
    return userpath / "Documents" / "NCAN" / "patients" / "fsaverage_wb" / "segmentation"


def build_group_atlas(subjects: Sequence[str], atlas_res: str, bipolar: bool) -> groupAtlas:
    atlas = Atlas.fs_LR_from_fsav(atlas_res)
    atlas.pt_sphere_name = "sphere.reg.surf.gii"
    flatbrain = groupAtlas(
        subjects=list(subjects),
        template_dir=get_template_segmentation_dir(),
        atlas=atlas,
    )
    if not bipolar:
        flatbrain._electrode_library = flatbrain._load_group_electrodes(bipolar=False)
    return flatbrain


def load_group_scan_overlay(flatbrain: groupAtlas) -> None:
    scan_key = "HCP_SCAN"
    flatbrain.load_cifti_data(get_scan_overlay_path(), scan_key)
    flatbrain.update_additional_ROI_value_map(scan_key, SCAN_VMAP, SCAN_CMAP)


def build_subject_colors(subjects: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors: dict[str, tuple[float, float, float, float]] = {}
    for idx, subject in enumerate(subjects):
        colors[subject] = cmap(idx % cmap.N)
    return colors


def build_flat_figure() -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    return fig, np.asarray(axes)


def add_subject_legend(
    ax: Axes,
    subject_colors: dict[str, tuple[float, float, float, float]],
) -> None:
    handles = [
        mpatches.Patch(color=color, label=subject)
        for subject, color in subject_colors.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title="Subject",
    )


def plot_flatmaps(
    flatbrain: groupAtlas,
    subject_colors: dict[str, tuple[float, float, float, float]],
    point_size: float,
) -> Figure:
    fig, axes = build_flat_figure()
    scan_key = "HCP_SCAN"
    sides = [("L", "lh"), ("R", "rh")]

    for ax, (side, hemi) in zip(axes, sides):
        hemi_df = flatbrain.electrode_library.loc[
            flatbrain.electrode_library["hemi"] == hemi
        ].copy()
        colors = hemi_df.reset_index()["subject"].map(subject_colors).to_list()
        subset = ["names", hemi_df.index]
        flatbrain.flatmap_plot(side, annot=False, outline=False, ax=ax)
        flatbrain.flatplot_additional_ROI(scan_key, side, ax=ax, opacity=0.45)
        flatbrain.flatplot_additional_ROI(scan_key, side, ax=ax, outline=True)
        flatbrain.plot_electrodes(
            ax=ax,
            hemi=hemi,
            bipolar=True,
            subset=subset,
            color=colors,
            size=point_size,
            alphas=[0.9] * len(hemi_df),
        )
        ax.set_title(f"SCAN electrodes {side}")
    
    
        flatbrain.fit_image_to_electrode(ax, flatbrain.electrode_library)

    add_subject_legend(axes[-1], subject_colors)
    fig.tight_layout()
    return fig


def load_surface_vertices(surface_path: Path) -> np.ndarray:
    surf = nib.load(str(surface_path))
    return np.asarray(surf.agg_data("pointset"))


def add_surface_electrodes(
    plotter: pv.Plotter,
    vertices: np.ndarray,
    hemi_df: pd.DataFrame,
    subject_colors: dict[str, tuple[float, float, float, float]],
    point_size: float,
) -> None:
    if hemi_df.empty:
        return

    coords = vertices[hemi_df["vert"].to_numpy(dtype=int)]
    color_values = hemi_df.reset_index()["subject"].map(subject_colors).to_list()
    poly = pv.PolyData(coords)
    poly["rgba"] = np.asarray(color_values)
    plotter.add_mesh(
        poly,
        scalars="rgba",
        rgba=True,
        render_points_as_spheres=True,
        point_size=point_size,
    )

class ElectrodePlotter(pv.Plotter):
    def __init__(
        self,
        *args: object,
        plot_config: Sequence[PlotConfigEntry] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._plot_config: list[PlotConfigEntry] = list(
            plot_config or [("lh", 0, 0, "Left"), ("rh", 0, 1, "Right")]
        )
    @property
    def plot_config(self) -> list[PlotConfigEntry]:
        return self._plot_config
    @plot_config.setter
    def plot_config(self,config) -> None:
        test_shape = np.asarray(config)
        if test_shape.shape[1] != 4:
            raise ValueError(f"4 dimensions expected in config, got {test_shape.shape[1]}")
        out = [(row[0],int(row[1]),int(row[2]),row[3]) for row in config]
            
        self._plot_config = out

    def swap_2_medial(self) -> str:
        for hemi, row_idx, col_idx, label in self.plot_config:
            self.subplot(row_idx, col_idx)
            if hemi == "lh":
                self.view_vector((1, 0, 0))
            else:
                self.view_vector((-1, 0, 0))
        return "medial"
    def swap_2_lateral(self) -> str:
        for hemi, row_idx, col_idx, label in self.plot_config:
            self.subplot(row_idx, col_idx)
            if hemi == "lh":
                self.view_vector((-1, 0, 0))
            else:
                self.view_vector((1, 0, 0))
        return "lateral"

    def swap_2_iso(self) -> str:
        iso_rot = 0.55
        iso_dip = 0.3
        for hemi, row_idx, col_idx, label in self.plot_config:
            self.subplot(row_idx, col_idx)
            if hemi == "lh":
                self.view_vector((-0.85, iso_rot, iso_dip))
            else:
                self.view_vector((0.85, iso_rot, iso_dip))
        return "iso"


def plot_surfaces(
    flatbrain: groupAtlas,
    subject_colors: dict[str, tuple[float, float, float, float]],
    point_size: float,
    interactive: bool,
) -> ElectrodePlotter:
    conte_surfs = get_conte_surfaces()
    flatbrain.set_plotter_surfaces(conte_surfs)
    vertices = {
        hemi: load_surface_vertices(surface_path)
        for hemi, surface_path in conte_surfs.items()
    }

    plotter = ElectrodePlotter(shape=(1, 2), window_size=(2700, 1080), off_screen=not interactive)
    scan_key = "HCP_SCAN"

    for hemi, row_idx, col_idx, label in plotter.plot_config:
        plotter.subplot(row_idx, col_idx)
        flatbrain.generic_surface_plot(hemi, ax=plotter)
        flatbrain.surfaceplot_additional_ROI(
            scan_key,
            hemi,
            ax=plotter,
            roi_opacity=0.35,
        )
        flatbrain.surfaceplot_additional_ROI(scan_key, hemi, ax=plotter, outline=True)
        hemi_df = flatbrain.electrode_library.loc[
            flatbrain.electrode_library["hemi"] == hemi
        ].copy()
        add_surface_electrodes(
            plotter=plotter,
            vertices=vertices[hemi],
            hemi_df=hemi_df,
            subject_colors=subject_colors,
            point_size=point_size,
        )
        plotter.add_text(f"SCAN electrodes {label}", font_size=14)
        if hemi == "lh":
            plotter.view_vector((-1, 0, 0))
        else:
            plotter.view_vector((1, 0, 0))

    return plotter


def save_flat_outputs(fig: Figure, outdir: Path) -> None:
    fig.savefig(outdir / "SCAN_all_electrodes_flatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "SCAN_all_electrodes_flatmaps.svg", bbox_inches="tight")


def save_surface_outputs(plotter: pv.Plotter, outdir: Path) -> None:
    plotter.screenshot(str(outdir / "SCAN_all_electrodes_surfaces.png"))


def main() -> None:
    args = parse_args()
    dataroot = get_dataroot()
    subjects_file = dataroot / "subjects.json"
    subjects = load_scan_subjects(subjects_file)
    outdir = args.outdir or dataroot / "group_figs" / "SCAN_all_electrodes"
    outdir.mkdir(parents=True, exist_ok=True)

    flatbrain = build_group_atlas(
        subjects=subjects,
        atlas_res=args.atlas_res,
        bipolar=args.bipolar,
    )
    flatbrain.export_electrode_summary(outdir,print2Latex=True)
    load_group_scan_overlay(flatbrain)
    subject_colors = build_subject_colors(subjects)

    flat_fig = plot_flatmaps(
        flatbrain=flatbrain,
        subject_colors=subject_colors,
        point_size=args.flat_size,
    )
    save_flat_outputs(flat_fig, outdir)

    surface_plotter = plot_surfaces(
        flatbrain=flatbrain,
        subject_colors=subject_colors,
        point_size=args.point_size,
        interactive=args.interactive,
    )
    plt.show(block=False)
    surface_plotter.save_graphic(outdir / "all_electrodes_surfaces_lateral.pdf")
    surface_plotter.save_graphic(outdir / "all_electrodes_surfaces_lateral.svg")
    format = surface_plotter.swap_2_medial()
    surface_plotter.save_graphic(outdir / f"all_electrodes_surfaces_{format}.pdf")
    surface_plotter.save_graphic(outdir / f"all_electrodes_surfaces_{format}.svg")
    save_surface_outputs(surface_plotter, outdir)
    format = surface_plotter.swap_2_iso()
    surface_plotter.save_graphic(outdir / f"all_electrodes_surfaces_{format}.pdf")
    surface_plotter.save_graphic(outdir / f"all_electrodes_surfaces_{format}.svg")
    if args.interactive:
        surface_plotter.show()
    else:
        surface_plotter.close()
    plt.close(flat_fig)

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()
