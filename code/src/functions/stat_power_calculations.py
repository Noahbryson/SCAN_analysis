from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from platform import system
from typing import Any
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.formula.api import mixedlm


class motor_mapping_power:
      """
      Estimate Aim 1 stability and approximate power for the group-level spatial
      concordance analysis between the electrophysiology-derived inter-effector map
      and the SCAN inter-effector reference map.

      The implementation follows a leave-one-patient-out jackknife workflow:
      1. Construct the full group electrophysiology map from the current cohort.
      2. Measure observed Dice overlap with the fixed SCAN map.
      3. Estimate the spatial null expectation and empirical p-value from the spin
         distribution.
      4. Reconstruct the group map repeatedly with one patient omitted at a time.
      5. Use jackknife variance across leave-one-out Dice values to estimate
         patient-driven uncertainty in the group-level overlap statistic.
      6. Convert that uncertainty into an approximate analytical power curve for
         candidate future sample sizes.
      """
      SCAN_VALUE_MAP: dict[float | int, str] = {
            0: "na",
            1.5: "inter",
            10: "hand",
            11: "face",
            17: "foot",
      }
      SCAN_COLOR_MAP: dict[str, tuple[float, float, float, float]] = {
            "na": (0.0, 0.0, 0.0, 0.0),
            "inter": (158 / 255, 38 / 255, 108 / 255, 1.0),
            "hand": (68 / 255, 1.0, 1.0, 1.0),
            "face": (1.0, 142 / 255, 52 / 255, 1.0),
            "foot": (32 / 255, 133 / 255, 44 / 255, 1.0),
      }
      MIXED_CLASS_MAP: dict[str, str] = {
            "hand-foot": "inter",
            "hand-face": "inter",
            "foot-face": "inter",
      }

      def __init__(self) -> None:
            pass

      def _resolve_default_paths(
            self,
            dataroot: str | Path | None,
            subjects_file: str | Path | None,
            segmentation_dir: str | Path | None,
            scan_map_path: str | Path | None,
            subject: str,
      ) -> tuple[Path, Path, Path, Path]:
            """Resolve default on-disk locations for SCAN data, subjects, surfaces, and SCAN map."""
            userpath = Path(os.path.expanduser("~"))
            boxpath = userpath if system() == "Windows" else userpath / "Library/CloudStorage/Box-Box"
            resolved_dataroot = Path(dataroot) if dataroot is not None else boxpath / "Brunner Lab" / "DATA" / "SCAN_Mayo"
            resolved_subjects_file = Path(subjects_file) if subjects_file is not None else resolved_dataroot / "subjects.json"
            resolved_segmentation = (
                  Path(segmentation_dir)
                  if segmentation_dir is not None
                  else userpath / "Documents" / "NCAN" / "patients" / subject / "segmentation"
            )
            resolved_scan_map = (
                  Path(scan_map_path)
                  if scan_map_path is not None
                  else boxpath / "Brunner Lab" / "DATA" / "SCAN_Mayo" / "imaging" / "HCP_Spots_Effectors_CS.dtseries.nii"
            )
            return resolved_dataroot, resolved_subjects_file, resolved_segmentation, resolved_scan_map

      def _build_motor_roi_set(
            self,
            electrode_library: pd.DataFrame,
            include_motor_rois: bool,
            include_insula: bool,
            include_operculum: bool,
      ) -> set[str]:
            """Build the ROI inclusion set used to restrict Aim 1 electrodes to planned motor territories."""
            if not include_motor_rois:
                  return set(electrode_library["region"].dropna().astype(str).tolist())

            rois = sorted(
                  set(
                        region
                        for region in electrode_library["region"].dropna().astype(str)
                        if "central" in region.lower()
                  )
            )
            insula_rois = sorted(
                  set(
                        region
                        for region in electrode_library["region"].dropna().astype(str)
                        if "insula" in region.lower() or "ins_ig" in region.lower()
                  )
            )
            if include_insula:
                  rois.extend(insula_rois)
                  rois.append("G_insular_short")
            if include_operculum:
                  rois.append("G_front_inf-Opercular")
            return set(rois)

      def _prepare_electrode_dataframe(
            self,
            task_power: pd.DataFrame,
            electrode_library: pd.DataFrame,
            include_motor_rois: bool,
            include_insula: bool,
            include_operculum: bool,
      ) -> pd.DataFrame:
            """Join task-power classifications to group-projected electrodes and apply Aim 1 ROI filtering."""
            
            temp = task_power.reset_index()
            temp['unique_session'] = temp['subject'] +'_'+ temp['session']
            self.hash_ID = {i:j for j,i in enumerate(np.unique(temp['unique_session']))}
            temp['hash'] = temp['unique_session'].map(self.hash_ID)
            temp.set_index(['subject','channel'],inplace=True)
            
            result = temp.join(electrode_library).reset_index()
            result = result.dropna(subset=["dist"]).copy()
            result["class"] = result["class"].replace(self.MIXED_CLASS_MAP)

            
            roi_set = self._build_motor_roi_set(
                  electrode_library=electrode_library.reset_index(),
                  include_motor_rois=include_motor_rois,
                  include_insula=include_insula,
                  include_operculum=include_operculum,
            )
            
            if include_motor_rois:
                  result = result.loc[result["region"].isin(roi_set)].copy()

            result["vert"] = result["vert"].astype(int)
            result["channel"] = result["channel"].astype(str)
            result["subject"] = result["subject"].astype(str)
            result["hemi"] = result["hemi"].astype(str)
            result["region"] = result["region"].astype(str)
            n_subjects = len(result['hash'].unique())
            return result,n_subjects

      def _load_surface_vertices(self, flatbrain: Any) -> dict[str, np.ndarray]:
            """Load atlas midthickness vertex coordinates for each hemisphere."""
            from PyBrain.modules.surface_projection import load_nii_file

            surface_vertices: dict[str, np.ndarray] = {}
            for hemi in ("lh", "rh"):
                  fp = flatbrain.midthickness_template_path.parent / flatbrain.midthickness_template_path.name.replace("[hemi]", hemi)
                  surface_vertices[hemi] = np.asarray(load_nii_file(fp).agg_data("pointset"), dtype=float)
            return surface_vertices

      def _build_reference_context(
            self,
            dataroot: str | Path | None,
            subjects_file: str | Path | None,
            segmentation_dir: str | Path | None,
            scan_map_path: str | Path | None,
            metric_name: str,
            atlas_res: str,
            subject: str,
            scan_key: str,
            include_motor_rois: bool,
            include_insula: bool,
            include_operculum: bool,
            neighborhood_radius: float,
            target_label: str,
      ) -> dict[str, Any]:
            """Assemble the reusable Aim 1 reference objects, maps, electrode table, and spin-cache path."""
            from src.SCAN_group_analysis import SCAN_group_analysis
            from PyBrain.modules.surface_projection import Atlas, groupAtlas

            resolved_dataroot, resolved_subjects_file, resolved_segmentation, resolved_scan_map = self._resolve_default_paths(
                  dataroot=dataroot,
                  subjects_file=subjects_file,
                  segmentation_dir=segmentation_dir,
                  scan_map_path=scan_map_path,
                  subject=subject,
            )
            atlas = Atlas.fs_LR_from_fsav(atlas_res)
            atlas.pt_sphere_name = "sphere.reg.surf.gii"

            analysis = SCAN_group_analysis(resolved_dataroot, resolved_subjects_file)
            flatbrain = groupAtlas(subjects=analysis.subjects, template_dir=resolved_segmentation, atlas=atlas)
            flatbrain.load_cifti_data(resolved_scan_map,     scan_key)
            flatbrain.update_additional_ROI_value_map(scan_key, dict(self.SCAN_VALUE_MAP), dict(self.SCAN_COLOR_MAP))

            task_power = analysis.load_task_power(metric_name=metric_name)
            electrode_data,n_subjects = self._prepare_electrode_dataframe(
                  task_power=task_power,
                  electrode_library=flatbrain.electrode_library,
                  include_motor_rois=include_motor_rois,
                  include_insula=include_insula,
                  include_operculum=include_operculum,
            )
            spin_cache_path = (
                  resolved_segmentation.parent
                  / "gifti"
                  / "spintest"
                  / "motor_maps"
                  / f"{neighborhood_radius}mm_radius"
                  / f"aim1_{target_label}"
            )

            return {
                  "flatbrain": flatbrain,
                  "electrode_data": electrode_data,
                  "scan_map": flatbrain.return_ROI_mapping(scan_key),
                  "spin_cache_path": spin_cache_path,
                  "n_subjects": n_subjects
            }

      def _build_group_ephys_map(
            self,
            flatbrain: Any,
            electrode_data: pd.DataFrame,
            neighborhood_radius: float,
            sigma: float,
            roi_key: str,
      ) -> dict[str, np.ndarray]:
            """Reconstruct a group electrophysiology map from the supplied electrode set."""
            if electrode_data.empty:
                  raise ValueError("Electrode dataset is empty.")

            voting_input = electrode_data.copy()
            voting_input["class"] = voting_input["class"].replace(self.MIXED_CLASS_MAP)
            _, voting_labels, label_id_map = flatbrain.euclidean_neighborhood_voting_map(
                  voting_input,
                  r=neighborhood_radius,
                  sigma=sigma,
            )
            flatbrain.add_additional_ROI_value_map(
                  roi_key,
                  voting_labels,
                  label_id_map,
                  overwrite=True,
            )
            return flatbrain.return_ROI_mapping(roi_key)

      def _combine_target_locus_arrays(
            self,
            experimental_map: dict[str, np.ndarray],
            reference_map: dict[str, np.ndarray],
            target_label: str,
      ) -> tuple[np.ndarray, np.ndarray]:
            """Restrict two hemisphere maps to the tested target locus and concatenate hemispheres."""
            from PyBrain.modules.statistics import test_locus_mask

            masked_experimental, masked_reference, _ = test_locus_mask(
                  experimental_map,
                  reference_map,
                  [target_label],
            )
            shared_hemis = [hemi for hemi in ("lh", "rh") if hemi in masked_experimental and hemi in masked_reference]
            if len(shared_hemis) == 0:
                  raise ValueError("No shared hemispheres are available for Aim 1 overlap calculation.")
            experimental = np.concatenate([np.asarray(masked_experimental[hemi], dtype=object) for hemi in shared_hemis])
            reference = np.concatenate([np.asarray(masked_reference[hemi], dtype=object) for hemi in shared_hemis])
            return experimental, reference

      def _compute_combined_dice(
            self,
            experimental_map: dict[str, np.ndarray],
            reference_map: dict[str, np.ndarray],
            target_label: str,
      ) -> float:
            """Compute one combined Dice statistic across all available hemispheres."""
            from PyBrain.modules.statistics import dice_coefficient

            experimental, reference = self._combine_target_locus_arrays(
                  experimental_map=experimental_map,
                  reference_map=reference_map,
                  target_label=target_label,
            )
            return float(dice_coefficient(experimental, reference, positive_label=target_label))

      def _compute_spin_null_summary(
            self,
            flatbrain: Any,
            ephys_map: dict[str, np.ndarray],
            scan_map: dict[str, np.ndarray],
            target_label: str,
            n_spins: int,
            spin_cache_path: Path,
            regenerate_spins: bool,
            ignore_legacy_spins: bool,
      ) -> dict[str, float | np.ndarray]:
            """Compute the observed Dice, spin-null expectation, and empirical one-sided p-value."""
            from PyBrain.modules.spin_nulls import SpinNullModel
            spin_model = SpinNullModel(
                  flatbrain.atlas,
                  ephys_map,
                  scan_map,
                  ignore_legacy_spins=ignore_legacy_spins,
            )
            observed_dice = self._compute_combined_dice(
                  experimental_map=ephys_map,
                  reference_map=scan_map,
                  target_label=target_label,
            )
            if regenerate_spins:
                  spin_model.generate_spins(n_spins, overwrite=True, spin_cache_path=spin_cache_path)
            else:
                  spin_model.generate_spins(n_spins, overwrite=False, spin_cache_path=spin_cache_path)
            n_eval, spins_to_use = spin_model._iter_spins(n_spins, use_stored_spins=True)
            null_distribution = np.zeros(n_eval, dtype=float)
            for idx, spun_map in enumerate(spins_to_use):
                  null_distribution[idx] = self._compute_combined_dice(
                        experimental_map=spun_map,
                        reference_map=scan_map,
                        target_label=target_label,
                  )
            p_spin = float((1 + np.sum(null_distribution >= observed_dice)) / (n_eval + 1))
            return {
                  "observed_dice": float(observed_dice),
                  "null_mean_dice": float(np.mean(null_distribution)),
                  "null_median_dice": float(np.median(null_distribution)),
                  "p_spin": p_spin,
                  "null_distribution": null_distribution,
            }

      def _compute_leave_one_out_dice(
            self,
            flatbrain: Any,
            electrode_data: pd.DataFrame,
            scan_map: dict[str, np.ndarray],
            neighborhood_radius: float,
            sigma: float,
            target_label: str,
      ) -> dict[str, float]:
            """Rebuild the group map with each patient omitted once and measure Dice overlap."""
            hash_ids = sorted(electrode_data["hash"].unique().tolist())
            reversed_hash = {j:i for i,j in self.hash_ID.items()}
            loo_dice: dict[str, float] = {}
            for hash_id in hash_ids:
                  subject_label = reversed_hash[hash_id]
                  print(f'leaving out {subject_label}')
                  subset = electrode_data.loc[electrode_data["hash"] != hash_id].copy()
                  loo_map = self._build_group_ephys_map(
                        flatbrain=flatbrain,
                        electrode_data=subset,
                        neighborhood_radius=neighborhood_radius,
                        sigma=sigma,
                        roi_key=f"aim1_loo_{subject_label}",
                  )
                  loo_dice[subject_label] = self._compute_combined_dice(
                        experimental_map=loo_map,
                        reference_map=scan_map,
                        target_label=target_label,
                  )
            return loo_dice

      def _compute_jackknife_variance(self, loo_dice_values: np.ndarray, n_patients: int) -> tuple[float, float, float]:
            """Compute mean leave-one-out Dice, jackknife variance, and jackknife standard error."""
            if n_patients <= 1:
                  raise ValueError("Jackknife variance requires at least 2 patients.")
            mean_loo = float(np.mean(loo_dice_values))
            variance = float(((n_patients - 1) / n_patients) * np.sum((loo_dice_values - mean_loo) ** 2))
            se = float(np.sqrt(max(variance, 0.0)))
            return mean_loo, variance, se

      def _approximate_power_at_n(
            self,
            delta_dice: float,
            jackknife_se: float,
            current_n: int,
            target_n: int,
            alpha: float,
      ) -> tuple[float, float]:
            """Approximate the expected standardized statistic and one-sided power at a candidate sample size."""
            if target_n <= 0:
                  raise ValueError("target_n must be > 0.")
            if jackknife_se < 0.0:
                  raise ValueError("jackknife_se must be >= 0.")
            if jackknife_se == 0.0:
                  if delta_dice > 0.0:
                        return float(np.inf), 1.0
                  if delta_dice < 0.0:
                        return float(-np.inf), 0.0
                  return 0.0, float(alpha)

            scaled_se = float(jackknife_se * np.sqrt(current_n / target_n))
            z_value = float(delta_dice / scaled_se)
            z_crit = float(norm.ppf(1.0 - alpha))
            power = float(norm.cdf(z_value - z_crit))
            return z_value, power

      def _required_n_analytic(
            self,
            delta_dice: float,
            jackknife_se: float,
            current_n: int,
            alpha: float,
            target_power: float,
      ) -> float:
            """Estimate the analytical sample size required to reach the requested power target."""
            z_crit = float(norm.ppf(1.0 - alpha))
            z_target = float(norm.ppf(target_power))
            if delta_dice <= 0.0:
                  return np.nan
            if jackknife_se == 0.0:
                  return float(current_n)
            return float(np.ceil(current_n * (((z_crit + z_target) * jackknife_se) / delta_dice) ** 2))

      def estimate_aim1_jackknife_variance(
            self,
            dataroot: str | Path | None = None,
            subjects_file: str | Path | None = None,
            segmentation_dir: str | Path | None = None,
            scan_map_path: str | Path | None = None,
            metric_name: str = "r-sq",
            atlas_res: str = "32k",
            subject: str = "fsaverage_wb",
            scan_key: str = "HCP_SCAN",
            neighborhood_radius: float = 6.0,
            sigma: float = 2.5,
            target_label: str = "inter",
            n_spins: int = 1000,
            include_motor_rois: bool = True,
            include_insula: bool = 0,
            include_operculum: bool = 0,
            regenerate_spins: bool = False,
            ignore_legacy_spins: bool = True,
            random_seed: int | None = None,
      ) -> pd.DataFrame:
            """
            Estimate Aim 1 leave-one-out jackknife variance for the current cohort.

            The workflow is:
            1. Build the full group electrophysiology map from the current cohort.
            2. Compute the observed Dice overlap with the fixed SCAN map.
            3. Estimate the spatial null expectation and empirical p-value from the spin
            null distribution.
            4. Rebuild the electrophysiology map once per leave-one-patient-out subset.
            5. Compute jackknife variance and standard error across the resulting Dice
            values.

            Args:
                  dataroot:
                        Root directory containing the SCAN cohort analysis outputs.
                  subjects_file:
                        Path to the `subjects.json` file used to initialize the cohort loader.
                  segmentation_dir:
                        Template segmentation directory used to initialize the group atlas.
                  scan_map_path:
                        Path to the imaging-derived SCAN CIFTI map.
                  metric_name:
                        Task-power metric to load from the existing analysis outputs.
                  atlas_res:
                        Surface atlas mesh resolution used to initialize the atlas and spin model.
                  subject:
                        Template subject name used when resolving the default segmentation path.
                  scan_key:
                        Name assigned to the SCAN map within the underlying `groupAtlas`.
                  neighborhood_radius:
                        Radius in millimeters used during group-map neighborhood voting.
                  sigma:
                        Gaussian distance-weighting width used during group-map construction.
                  target_label:
                        Electrophysiology/SCAN class label whose overlap is tested.
                  n_spins:
                        Number of spherical spin permutations used to estimate the null
                        distribution for the observed group map.
                  alpha:
                        One-sided significance threshold used in the analytical power projection.
                  target_power:
                        Target prospective power level used when reporting the estimated required
                        sample size.
                  include_motor_rois:
                        If `True`, restrict electrodes to the planned motor-territory ROI subset.
                  include_insula:
                        If `True`, include insular ROIs when `include_motor_rois` is enabled.
                  include_operculum:
                        If `True`, include opercular ROIs when `include_motor_rois` is enabled.
                  regenerate_spins:
                        If `True`, regenerate the cached spin permutations.
                  ignore_legacy_spins:
                        If `True`, ignore legacy value-based spin caches.
                  random_seed:
                        Present for API consistency. The jackknife workflow itself is deterministic
                        when cached spins are reused.

            Returns:
                  A one-row dataframe containing the observed group overlap, null expectation,
                  empirical spin p-value, leave-one-out Dice values, and jackknife variance
                  summary for the current cohort.
            """
            _ = random_seed
            if n_spins <= 0:
                  raise ValueError("n_spins must be > 0.")

            context = self._build_reference_context(
                  dataroot=dataroot,
                  subjects_file=subjects_file,
                  segmentation_dir=segmentation_dir,
                  scan_map_path=scan_map_path,
                  metric_name=metric_name,
                  atlas_res=atlas_res,
                  subject=subject,
                  scan_key=scan_key,
                  include_motor_rois=include_motor_rois,
                  include_insula=include_insula,
                  include_operculum=include_operculum,
                  neighborhood_radius=neighborhood_radius,
                  target_label=target_label,
            )
            current_n = context['n_subjects']

            full_map = self._build_group_ephys_map(
                  flatbrain=context["flatbrain"],
                  electrode_data=context["electrode_data"],
                  neighborhood_radius=neighborhood_radius,
                  sigma=sigma,
                  roi_key="aim1_full_group_map",
            )
            spin_summary = self._compute_spin_null_summary(
                  flatbrain=context["flatbrain"],
                  ephys_map=full_map,
                  scan_map=context["scan_map"],
                  target_label=target_label,
                  n_spins=n_spins,
                  spin_cache_path=context["spin_cache_path"],
                  regenerate_spins=regenerate_spins,
                  ignore_legacy_spins=ignore_legacy_spins,
            )
            loo_dice_by_subject = self._compute_leave_one_out_dice(
                  flatbrain=context["flatbrain"],
                  electrode_data=context["electrode_data"],
                  scan_map=context["scan_map"],
                  neighborhood_radius=neighborhood_radius,
                  sigma=sigma,
                  target_label=target_label,
            )
            loo_subjects = sorted(loo_dice_by_subject.keys())
            loo_values = np.asarray([loo_dice_by_subject[subject] for subject in loo_subjects], dtype=float)
            mean_loo_dice, jackknife_variance, jackknife_se = self._compute_jackknife_variance(
                  loo_dice_values=loo_values,
                  n_patients=current_n,
            )
            delta_dice = float(spin_summary["observed_dice"] - spin_summary["null_mean_dice"])
            if jackknife_se == 0.0:
                  if delta_dice > 0.0:
                        z_approx = float(np.inf)
                  elif delta_dice < 0.0:
                        z_approx = float(-np.inf)
                  else:
                        z_approx = 0.0
            else:
                  z_approx = float(delta_dice / jackknife_se)

            row: dict[str, float | int | str] = {
                  "current_n_patients": int(current_n),
                  "target_label": str(target_label),
                  "n_spins": int(n_spins),
                  "observed_dice": float(spin_summary["observed_dice"]),
                  "null_mean_dice": float(spin_summary["null_mean_dice"]),
                  "null_median_dice": float(spin_summary["null_median_dice"]),
                  "p_spin": float(spin_summary["p_spin"]),
                  "mean_leave_one_out_dice": float(mean_loo_dice),
                  "jackknife_variance": float(jackknife_variance),
                  "jackknife_se": float(jackknife_se),
                  "delta_dice": float(delta_dice),
                  "z_approx": float(z_approx),
                  "n_leave_one_out_maps": int(len(loo_subjects)),
            }
            for subject in loo_subjects:
                  row[f"loo_dice_{subject}"] = float(loo_dice_by_subject[subject])
            return pd.DataFrame([row])

      def calculate_aim1_power_curve(
            self,
            variance_summary: pd.DataFrame | pd.Series | dict[str, Any],
            patient_counts: int | Sequence[int],
            alpha: float = 0.05,
            target_power: float = 0.80,
      ) -> pd.DataFrame:
            """
            Calculate an approximate Aim 1 analytical power curve from a previously
            estimated jackknife variance summary.

            Args:
                  variance_summary:
                        One-row summary from `estimate_aim1_jackknife_variance`, or an
                        equivalent mapping/series containing at least `current_n_patients`,
                        `observed_dice`, `null_mean_dice`, and `jackknife_se`.
                  patient_counts:
                        Candidate patient sample sizes to evaluate. If the current cohort
                        size is not included, it is inserted automatically.
                  alpha:
                        One-sided significance threshold used in the analytical power projection.
                  target_power:
                        Target prospective power level used when reporting the estimated
                        required sample size.

            Returns:
                  A dataframe with one row per candidate patient count containing the
                  projected standardized statistic and approximate analytical power.
            """
            if isinstance(patient_counts, int):
                  candidate_patient_counts = [int(patient_counts)]
            else:
                  candidate_patient_counts = [int(value) for value in patient_counts]
            if len(candidate_patient_counts) == 0:
                  raise ValueError("At least one patient count must be provided.")
            if any(value <= 0 for value in candidate_patient_counts):
                  raise ValueError("All patient counts must be > 0.")
            if not 0.0 < alpha < 1.0:
                  raise ValueError("alpha must be in (0, 1).")
            if not 0.0 < target_power < 1.0:
                  raise ValueError("target_power must be in (0, 1).")

            if isinstance(variance_summary, pd.DataFrame):
                  if variance_summary.empty:
                        raise ValueError("variance_summary dataframe cannot be empty.")
                  summary = variance_summary.iloc[0].to_dict()
            elif isinstance(variance_summary, pd.Series):
                  summary = variance_summary.to_dict()
            else:
                  summary = dict(variance_summary)

            required_keys = {"current_n_patients", "observed_dice", "null_mean_dice", "jackknife_se"}
            missing_keys = required_keys.difference(summary.keys())
            if len(missing_keys) > 0:
                  raise KeyError(f"variance_summary is missing required keys: {sorted(missing_keys)}")

            current_n = int(summary["current_n_patients"])
            if any(value < current_n for value in candidate_patient_counts):
                  raise ValueError(f"All candidate patient counts must be >= the current cohort size ({current_n}).")
            candidate_patient_counts = sorted(set(candidate_patient_counts + [current_n]))

            observed_dice = float(summary["observed_dice"])
            null_mean_dice = float(summary["null_mean_dice"])
            jackknife_se = float(summary["jackknife_se"])
            delta_dice = float(summary.get("delta_dice", observed_dice - null_mean_dice))
            z_approx = float(summary.get("z_approx", np.inf if jackknife_se == 0.0 and delta_dice > 0.0 else (-np.inf if jackknife_se == 0.0 and delta_dice < 0.0 else (0.0 if jackknife_se == 0.0 else delta_dice / jackknife_se))))
            required_n_analytic = self._required_n_analytic(
                  delta_dice=delta_dice,
                  jackknife_se=jackknife_se,
                  current_n=current_n,
                  alpha=alpha,
                  target_power=target_power,
            )

            rows: list[dict[str, float | int]] = []
            for n_patients in candidate_patient_counts:
                  expected_z, approximate_power = self._approximate_power_at_n(
                        delta_dice=delta_dice,
                        jackknife_se=jackknife_se,
                        current_n=current_n,
                        target_n=n_patients,
                        alpha=alpha,
                  )
                  rows.append(
                        {
                              "n_patients": int(n_patients),
                              "current_n_patients": int(current_n),
                              "alpha": float(alpha),
                              "target_power": float(target_power),
                              "observed_dice": float(observed_dice),
                              "null_mean_dice": float(null_mean_dice),
                              "jackknife_se": float(jackknife_se),
                              "delta_dice": float(delta_dice),
                              "z_approx": float(z_approx),
                              "z_expected": float(expected_z),
                              "approx_power": float(approximate_power),
                              "required_n_analytic": float(required_n_analytic) if not np.isnan(required_n_analytic) else np.nan,
                              "required_n_from_candidates": np.nan,
                        }
                  )

            result = pd.DataFrame(rows)
            reaching_target = result.loc[result["approx_power"] >= target_power, "n_patients"]
            required_n_from_candidates = float(reaching_target.iloc[0]) if len(reaching_target) > 0 else np.nan
            result["required_n_from_candidates"] = required_n_from_candidates
            return result

      def estimate_aim1_jackknife_power(
            self,
            patient_counts: int | Sequence[int],
            dataroot: str | Path | None = None,
            subjects_file: str | Path | None = None,
            segmentation_dir: str | Path | None = None,
            scan_map_path: str | Path | None = None,
            metric_name: str = "r-sq",
            atlas_res: str = "32k",
            subject: str = "fsaverage_wb",
            scan_key: str = "HCP_SCAN",
            neighborhood_radius: float = 6.0,
            sigma: float = 2.5,
            target_label: str = "inter",
            n_spins: int = 1000,
            alpha: float = 0.05,
            target_power: float = 0.80,
            include_motor_rois: bool = True,
            include_insula: bool = 0,
            include_operculum: bool = 0,
            regenerate_spins: bool = False,
            ignore_legacy_spins: bool = True,
            random_seed: int | None = None,
            return_simulation_summary: bool = False,
      ) -> pd.DataFrame:
            """Convenience wrapper that estimates jackknife variance once and then calculates the power curve."""
            variance_summary = self.estimate_aim1_jackknife_variance(
                  dataroot=dataroot,
                  subjects_file=subjects_file,
                  segmentation_dir=segmentation_dir,
                  scan_map_path=scan_map_path,
                  metric_name=metric_name,
                  atlas_res=atlas_res,
                  subject=subject,
                  scan_key=scan_key,
                  neighborhood_radius=neighborhood_radius,
                  sigma=sigma,
                  target_label=target_label,
                  n_spins=n_spins,
                  include_motor_rois=include_motor_rois,
                  include_insula=include_insula,
                  include_operculum=include_operculum,
                  regenerate_spins=regenerate_spins,
                  ignore_legacy_spins=ignore_legacy_spins,
                  random_seed=random_seed,
            )
            power_curve = self.calculate_aim1_power_curve(
                  variance_summary=variance_summary,
                  patient_counts=patient_counts,
                  alpha=alpha,
                  target_power=target_power,
            )
            if return_simulation_summary:
                  summary_columns = [column for column in variance_summary.columns if column not in power_curve.columns]
                  for column in summary_columns:
                        power_curve[column] = variance_summary.iloc[0][column]
            return power_curve

      def simulate_aim1_spatial_power(
            self,
            patient_counts: int | Sequence[int],
            dataroot: str | Path | None = None,
            subjects_file: str | Path | None = None,
            segmentation_dir: str | Path | None = None,
            scan_map_path: str | Path | None = None,
            metric_name: str = "r-sq",
            atlas_res: str = "32k",
            subject: str = "fsaverage_wb",
            scan_key: str = "HCP_SCAN",
            neighborhood_radius: float = 6.0,
            sigma: float = 2.5,
            target_label: str = "inter",
            n_spins: int = 1000,
            n_simulations: int = 1000,
            alpha: float = 0.05,
            include_motor_rois: bool = True,
            include_insula: bool = True,
            include_operculum: bool = True,
            localization_sd_mm: float = 0.0,
            misclassification_rate: float = 0.0,
            exclusion_rate: float = 0.0,
            coverage_requirements: dict[str, Sequence[str]] | None = None,
            min_coverage_per_requirement: int = 1,
            regenerate_spins: bool = False,
            random_seed: int | None = None,
            return_simulation_summary: bool = False,
      ) -> pd.DataFrame:
            """Compatibility wrapper forwarding the old Aim 1 API to the jackknife-based implementation."""
            _ = n_simulations
            _ = localization_sd_mm
            _ = misclassification_rate
            _ = exclusion_rate
            _ = coverage_requirements
            _ = min_coverage_per_requirement
            variance_summary = self.estimate_aim1_jackknife_variance(
                  dataroot=dataroot,
                  subjects_file=subjects_file,
                  segmentation_dir=segmentation_dir,
                  scan_map_path=scan_map_path,
                  metric_name=metric_name,
                  atlas_res=atlas_res,
                  subject=subject,
                  scan_key=scan_key,
                  neighborhood_radius=neighborhood_radius,
                  sigma=sigma,
                  target_label=target_label,
                  n_spins=n_spins,
                  include_motor_rois=include_motor_rois,
                  include_insula=include_insula,
                  include_operculum=include_operculum,
                  regenerate_spins=regenerate_spins,
                  random_seed=random_seed,
            )
            power_curve = self.calculate_aim1_power_curve(
                  variance_summary=variance_summary,
                  patient_counts=patient_counts,
                  alpha=alpha,
                  target_power=0.80,
            )
            if return_simulation_summary:
                  summary_columns = [column for column in variance_summary.columns if column not in power_curve.columns]
                  for column in summary_columns:
                        power_curve[column] = variance_summary.iloc[0][column]
            return power_curve


class thalamocortical_LMM_power:
      def __init__(self) -> None:
            pass

      def _sample_count(
            self,
            spec: int | tuple[int, int] | Sequence[int],
            rng: np.random.Generator,
      ) -> int:
            if isinstance(spec, int):
                  return max(0, spec)
            if isinstance(spec, tuple) and len(spec) == 2:
                  low, high = spec
                  if high < low:
                        raise ValueError("Count range upper bound must be >= lower bound.")
                  return int(rng.integers(low, high + 1))
            if isinstance(spec, Sequence):
                  if len(spec) == 0:
                        raise ValueError("Count sequence cannot be empty.")
                  return int(rng.choice(np.asarray(spec, dtype=int)))
            raise TypeError("Count specification must be an int, a length-2 tuple, or a non-empty sequence.")

      def _sample_fraction(self, spec: float | tuple[float, float], rng: np.random.Generator) -> float:
            if isinstance(spec, (float, int)):
                  value = float(spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                  low, high = float(spec[0]), float(spec[1])
                  if high < low:
                        raise ValueError("Fraction range upper bound must be >= lower bound.")
                  value = float(rng.uniform(low, high))
            else:
                  raise TypeError("Fraction specification must be a float or a length-2 tuple.")
            if not 0.0 <= value < 1.0:
                  raise ValueError("Artifact proportion must be in [0, 1).")
            return value

      def _apply_artifact_dropout(self, count: int, artifact_fraction: float, rng: np.random.Generator) -> int:
            if count <= 0:
                  return 0
            keep_probability = 1.0 - artifact_fraction
            retained = int(rng.binomial(count, keep_probability))
            return retained

      def _trial_type_to_factors(self, trial_type: str) -> tuple[float, float]:
            mapping = {
                  "TP": (0.5, 0.5),
                  "TN": (-0.5, -0.5),
                  "FP": (-0.5, 0.5),
                  "FN": (0.5, -0.5),
            }
            if trial_type not in mapping:
                  raise ValueError(f"Unsupported trial type: {trial_type}")
            return mapping[trial_type]

      def _simulate_aim3_dataset(
            self,
            n_patients: int,
            motor_electrodes_per_patient: int | tuple[int, int] | Sequence[int],
            intereffector_electrodes_per_patient: int | tuple[int, int] | Sequence[int],
            tp_trials_per_patient: int | tuple[int, int] | Sequence[int],
            tn_trials_per_patient: int | tuple[int, int] | Sequence[int],
            fp_trials_per_patient: int | tuple[int, int] | Sequence[int],
            fn_trials_per_patient: int | tuple[int, int] | Sequence[int],
            artifact_fraction: float | tuple[float, float],
            intercept: float,
            beta_functional_class: float,
            beta_coherence: float,
            beta_motor_response: float,
            beta_functional_class_by_coherence: float,
            beta_functional_class_by_motor_response: float,
            beta_coherence_by_motor_response: float,
            beta_functional_class_by_coherence_by_motor_response: float,
            patient_sd: float,
            electrode_sd: float,
            residual_sd: float,
            rng: np.random.Generator,
      ) -> pd.DataFrame:
            rows: list[dict[str, Any]] = []
            trial_specs = {
                  "TP": tp_trials_per_patient,
                  "TN": tn_trials_per_patient,
                  "FP": fp_trials_per_patient,
                  "FN": fn_trials_per_patient,
            }
            artifact_value = self._sample_fraction(artifact_fraction, rng)

            for patient_idx in range(n_patients):
                  patient_label = f"patient_{patient_idx:03d}"
                  patient_effect = float(rng.normal(0.0, patient_sd))

                  class_specs = {
                        "motor": {
                              "code": 0.5,
                              "n_electrodes": self._sample_count(motor_electrodes_per_patient, rng),
                        },
                        "intereffector": {
                              "code": -0.5,
                              "n_electrodes": self._sample_count(intereffector_electrodes_per_patient, rng),
                        },
                  }

                  for class_name, class_info in class_specs.items():
                        functional_class = float(class_info["code"])
                        for electrode_idx in range(class_info["n_electrodes"]):
                              electrode_label = f"{patient_label}_{class_name}_{electrode_idx:03d}"
                              electrode_effect = float(rng.normal(0.0, electrode_sd))

                              for trial_type, trial_spec in trial_specs.items():
                                    coherence, motor_response = self._trial_type_to_factors(trial_type)
                                    planned_trials = self._sample_count(trial_spec, rng)
                                    retained_trials = self._apply_artifact_dropout(planned_trials, artifact_value, rng)
                                    for trial_idx in range(retained_trials):
                                          mean_response = (
                                                intercept
                                                + beta_functional_class * functional_class
                                                + beta_coherence * coherence
                                                + beta_motor_response * motor_response
                                                + beta_functional_class_by_coherence * functional_class * coherence
                                                + beta_functional_class_by_motor_response * functional_class * motor_response
                                                + beta_coherence_by_motor_response * coherence * motor_response
                                                + beta_functional_class_by_coherence_by_motor_response * functional_class * coherence * motor_response
                                                + patient_effect
                                                + electrode_effect
                                          )
                                          response = float(rng.normal(mean_response, residual_sd))
                                          rows.append(
                                                {
                                                      "patient": patient_label,
                                                      "electrode": electrode_label,
                                                      "functional_class": functional_class,
                                                      "cue_target_coherence": coherence,
                                                      "motor_response": motor_response,
                                                      "trial_type": trial_type,
                                                      "trial_index": trial_idx,
                                                      "bbg_response": response,
                                                }
                                          )

            return pd.DataFrame(rows)

      def _fit_aim3_mixedlm(self, simulated_data: pd.DataFrame) -> dict[str, float]:
            if simulated_data.empty:
                  raise ValueError("Simulated dataset is empty.")
            formula = "bbg_response ~ functional_class * cue_target_coherence * motor_response"
            model = mixedlm(
                  formula=formula,
                  data=simulated_data,
                  groups=simulated_data["patient"],
                  re_formula="1",
                  vc_formula={"electrode": "0 + C(electrode)"},
            )
            with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  result = model.fit(reml=False, method="lbfgs", disp=False)
            return {key: float(value) for key, value in result.pvalues.items()}

      def simulate_aim3_mixedlm_power(
            self,
            patient_counts: int | Sequence[int],
            motor_electrodes_per_patient: int | tuple[int, int] | Sequence[int],
            intereffector_electrodes_per_patient: int | tuple[int, int] | Sequence[int],
            tp_trials_per_patient: int | tuple[int, int] | Sequence[int],
            tn_trials_per_patient: int | tuple[int, int] | Sequence[int],
            fp_trials_per_patient: int | tuple[int, int] | Sequence[int],
            fn_trials_per_patient: int | tuple[int, int] | Sequence[int],
            artifact_fraction: float | tuple[float, float],
            intercept: float,
            beta_functional_class_by_coherence: float,
            beta_functional_class_by_motor_response: float,
            beta_functional_class_by_coherence_by_motor_response: float,
            beta_functional_class: float = 0.0,
            beta_coherence: float = 0.0,
            beta_motor_response: float = 0.0,
            beta_coherence_by_motor_response: float = 0.0,
            patient_sd: float = 1.0,
            electrode_sd: float = 1.0,
            residual_sd: float = 1.0,
            alpha: float = 0.05,
            multiple_test_correction: str | None = "bonferroni",
            n_simulations: int = 1000,
            random_seed: int | None = None,
            return_simulation_summary: bool = False,
      ) -> pd.DataFrame:
            """
            Estimate Aim 3 power for the planned linear mixed effects model by simulation.

      The fitted model is:
      `bbg_response ~ functional_class * cue_target_coherence * motor_response`
      with random intercepts for patient and electrode nested within patient.

      Factor coding:
      - `functional_class`: motor = 0.5, intereffector = -0.5
      - `cue_target_coherence`: coherent = 0.5, incoherent = -0.5
      - `motor_response`: response = 0.5, no response = -0.5

      Trial-type mapping:
      - `TP`: coherent + response
      - `TN`: incoherent + no response
      - `FP`: incoherent + response
      - `FN`: coherent + no response

      Returns one row per candidate patient count with estimated power for:
      - `beta_4`: functional class by coherence
      - `beta_5`: functional class by motor response
      - `beta_7`: three-way interaction

      Args:
            patient_counts:
                  Candidate patient sample sizes to evaluate. May be a single integer or a
                  sequence of integers. One output row is returned per candidate value.
            motor_electrodes_per_patient:
                  Planned number of motor-eloquent electrodes contributed by each patient.
                  Accepts a fixed integer, a `(min, max)` tuple sampled uniformly as integers,
                  or a sequence of possible counts sampled with equal probability.
            intereffector_electrodes_per_patient:
                  Planned number of inter-effector electrodes contributed by each patient.
                  Uses the same input conventions as `motor_electrodes_per_patient`.
            tp_trials_per_patient:
                  Number of usable true-positive trials per patient before artifact removal.
                  May be fixed or sampled using the same conventions as electrode counts.
                  `TP` is coded as coherent cue plus motor response.
            tn_trials_per_patient:
                  Number of usable true-negative trials per patient before artifact removal.
                  `TN` is coded as incoherent cue plus no motor response.
            fp_trials_per_patient:
                  Number of usable false-positive trials per patient before artifact removal.
                  `FP` is coded as incoherent cue plus motor response.
            fn_trials_per_patient:
                  Number of usable false-negative trials per patient before artifact removal.
                  `FN` is coded as coherent cue plus no motor response.
            artifact_fraction:
                  Expected proportion of planned trials lost to artifact or incomplete behavior.
                  May be a fixed fraction or a `(min, max)` tuple sampled uniformly once per
                  simulated dataset. The retained trial count is sampled with a binomial draw.
            intercept:
                  Grand mean BBG response when all coded predictors are at zero.
                  Because the design uses centered effect coding, this is the overall mean
                  across functional class, coherence, and motor-response conditions.
            beta_functional_class_by_coherence:
                  Expected effect size for the functional class by cue-target coherence
                  interaction (`beta_4`), the first primary power target.
            beta_functional_class_by_motor_response:
                  Expected effect size for the functional class by motor-response interaction
                  (`beta_5`), the second primary power target.
            beta_functional_class_by_coherence_by_motor_response:
                  Expected effect size for the functional class by coherence by motor-response
                  interaction (`beta_7`), the third primary power target.
            beta_functional_class:
                  Optional main effect of functional class. Motor sites are coded `0.5` and
                  inter-effector sites are coded `-0.5`.
            beta_coherence:
                  Optional main effect of cue-target coherence. Coherent trials are coded `0.5`
                  and incoherent trials are coded `-0.5`.
            beta_motor_response:
                  Optional main effect of motor response. Trials with a motor response are coded
                  `0.5` and trials without a response are coded `-0.5`.
            beta_coherence_by_motor_response:
                  Optional two-way interaction between coherence and motor response. This is not
                  one of the primary power targets but can be included in the data-generating model.
            patient_sd:
                  Standard deviation of the patient-level random intercept. Controls between-patient
                  variability in BBG response.
            electrode_sd:
                  Standard deviation of the electrode-level random intercept nested within patient.
                  Controls between-electrode variability within patient.
            residual_sd:
                  Standard deviation of the trial-level residual error term. Controls within-electrode
                  trial-to-trial variability after fixed and random effects are applied.
            alpha:
                  Nominal type-I error rate before any multiple-testing correction.
            multiple_test_correction:
                  Correction applied across the three primary interaction tests. Currently supports
                  `None` and `"bonferroni"`.
            n_simulations:
                  Number of simulated datasets generated and fit for each candidate patient count.
                  Larger values produce more stable power estimates at higher compute cost.
            random_seed:
                  Seed passed to `numpy.random.default_rng` for reproducible simulation results.
            return_simulation_summary:
                  If `True`, include summary information such as the mean number of simulated
                  observations per dataset in the returned table.

      Returns:
            A dataframe with one row per evaluated patient count. Columns include the candidate
            sample size, number of successful and failed model fits, alpha values used for testing,
            and estimated power for `beta_4`, `beta_5`, and `beta_7`.
      """
            if isinstance(patient_counts, int):
                  candidate_patient_counts = [patient_counts]
            else:
                  candidate_patient_counts = [int(value) for value in patient_counts]
            if len(candidate_patient_counts) == 0:
                  raise ValueError("At least one patient count must be provided.")
            if any(value <= 0 for value in candidate_patient_counts):
                  raise ValueError("All patient counts must be > 0.")
            if n_simulations <= 0:
                  raise ValueError("n_simulations must be > 0.")
            if alpha <= 0.0 or alpha >= 1.0:
                  raise ValueError("alpha must be in (0, 1).")

            corrected_alpha = alpha
            if multiple_test_correction is not None:
                  correction = multiple_test_correction.lower()
                  if correction == "bonferroni":
                        corrected_alpha = alpha / 3.0
                  else:
                        raise ValueError(f"Unsupported multiple test correction: {multiple_test_correction}")

            rng = np.random.default_rng(random_seed)
            interaction_terms = {
                  "beta_4": "functional_class:cue_target_coherence",
                  "beta_5": "functional_class:motor_response",
                  "beta_7": "functional_class:cue_target_coherence:motor_response",
            }
            results: list[dict[str, float | int]] = []

            for n_patients in candidate_patient_counts:
                  significant_counts = {label: 0 for label in interaction_terms}
                  converged_fits = 0
                  failed_fits = 0
                  mean_observations = 0.0

                  for _ in range(n_simulations):
                        simulated_data = self._simulate_aim3_dataset(
                              n_patients=n_patients,
                              motor_electrodes_per_patient=motor_electrodes_per_patient,
                              intereffector_electrodes_per_patient=intereffector_electrodes_per_patient,
                              tp_trials_per_patient=tp_trials_per_patient,
                              tn_trials_per_patient=tn_trials_per_patient,
                              fp_trials_per_patient=fp_trials_per_patient,
                              fn_trials_per_patient=fn_trials_per_patient,
                              artifact_fraction=artifact_fraction,
                              intercept=intercept,
                              beta_functional_class=beta_functional_class,
                              beta_coherence=beta_coherence,
                              beta_motor_response=beta_motor_response,
                              beta_functional_class_by_coherence=beta_functional_class_by_coherence,
                              beta_functional_class_by_motor_response=beta_functional_class_by_motor_response,
                              beta_coherence_by_motor_response=beta_coherence_by_motor_response,
                              beta_functional_class_by_coherence_by_motor_response=beta_functional_class_by_coherence_by_motor_response,
                              patient_sd=patient_sd,
                              electrode_sd=electrode_sd,
                              residual_sd=residual_sd,
                              rng=rng,
                        )
                        mean_observations += float(len(simulated_data))

                        try:
                              pvalues = self._fit_aim3_mixedlm(simulated_data)
                              converged_fits += 1
                        except Exception:
                              failed_fits += 1
                              continue

                        for power_label, term_name in interaction_terms.items():
                              if term_name in pvalues and pvalues[term_name] < corrected_alpha:
                                    significant_counts[power_label] += 1

                  denominator = max(converged_fits, 1)
                  result_row: dict[str, float | int] = {
                        "n_patients": int(n_patients),
                        "n_simulations": int(n_simulations),
                        "converged_fits": int(converged_fits),
                        "failed_fits": int(failed_fits),
                        "alpha": float(alpha),
                        "corrected_alpha": float(corrected_alpha),
                        "power_beta_4": float(significant_counts["beta_4"] / denominator),
                        "power_beta_5": float(significant_counts["beta_5"] / denominator),
                        "power_beta_7": float(significant_counts["beta_7"] / denominator),
                  }
                  if return_simulation_summary:
                        result_row["mean_observations"] = float(mean_observations / n_simulations)
                  results.append(result_row)

            return pd.DataFrame(results)
