import warnings
from pathlib import Path
import numpy as np
import scipy.stats as st
from sklearn import metrics
from typing import Iterable,Hashable,Optional,Sequence
import matplotlib.pyplot as plt


def _validate_1d_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Validate and coerce two 1-D arrays to matching flat NumPy arrays."""
      y_true_arr = np.asarray(y_true).ravel()
      y_pred_arr = np.asarray(y_pred).ravel()

      if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays.")
      if y_true_arr.size == 0 or y_pred_arr.size == 0:
            raise ValueError("Inputs must be non-empty.")
      if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            warnings.warn('Inputs are not equal length, be sure to use NMI in this case')

      return y_true_arr, y_pred_arr


def test_locus_mask(y_true: dict[Hashable, np.ndarray], y_pred: dict[Hashable, np.ndarray], target_values: Iterable[int | float | str] = (0,)) -> tuple[dict[Hashable, np.ndarray], dict[Hashable, np.ndarray], dict[Hashable, np.ndarray]]:
      """
      Filter two keyed maps to tested loci and return only maps with the target indices present, plus the masks.

      Args:
            y_true: Dictionary of reference 1-D vectors keyed by region/label.
            y_pred: Dictionary of comparison 1-D vectors keyed identically to `y_true`.
            target_values: Values of interest to preserve.
            
      Returns:
            y_true_out: Dictionary with non-locus values suppressed in each true map.
            y_pred_out: Dictionary with non-locus values suppressed in each pred map.
            masks: Dictionary of boolean tested-locus masks for each key.
      """
      if set(y_true.keys()) != set(y_pred.keys()):
            raise ValueError("Input dictionaries must have identical keys.")

      vals = list(target_values)
      y_true_out: dict[Hashable, np.ndarray] = {}
      y_pred_out: dict[Hashable, np.ndarray] = {}
      masks: dict[Hashable, np.ndarray] = {}

      for key in y_true:
            y_true_arr, y_pred_arr = _validate_1d_arrays(y_true[key], y_pred[key])
            vals_true = np.isin(y_true_arr, vals)
            vals_pred = np.isin(y_pred_arr, vals)
            
            locus = (vals_true) | (vals_pred)
            
            mask = locus
            true_map = y_true_arr[np.where(locus)[0]]
            pred_map = y_pred_arr[np.where(locus)[0]]
            
            y_true_out[key] = true_map
            y_pred_out[key] = pred_map
            masks[key] = mask

      return y_true_out, y_pred_out, masks


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, positive_label: Sequence|Hashable = 1) -> float:
      """Compute binary Dice coefficient for a chosen positive label."""
      y_true_arr, y_pred_arr = _validate_1d_arrays(y_true, y_pred)
      if isinstance(positive_label,(list,tuple,np.ndarray,set)):
            true_pos = np.isin(y_true_arr, positive_label)
            pred_pos = np.isin(y_pred_arr, positive_label)
      else:            
            true_pos = y_true_arr == positive_label
            pred_pos = y_pred_arr == positive_label

      intersection = np.sum(true_pos & pred_pos)
      denom = np.sum(true_pos) + np.sum(pred_pos)
      if denom == 0:
            return 1.0
      return float((2.0 * intersection) / denom)


def multi_class_dice(y_true: np.ndarray, y_pred: np.ndarray,labels:Optional[Sequence]=None) -> tuple[np.ndarray, np.ndarray, float]:
      """
      Compute multiclass Dice statistics.

      Returns:
            labels: sorted unique class labels from both arrays.
            class_dice: Dice score for each label in `labels`.
            macro_dice: unweighted mean of per-class Dice scores.
      """
      y_true_arr, y_pred_arr = _validate_1d_arrays(y_true, y_pred)
      if labels is None:
            labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))

      class_dice = np.zeros(np.shape(labels)[0], dtype=float)
      for idx, label in enumerate(labels):
            true_pos = y_true_arr == label
            pred_pos = y_pred_arr == label
            intersection = np.sum(true_pos & pred_pos)
            denom = np.sum(true_pos) + np.sum(pred_pos)
            class_dice[idx] = 1.0 if denom == 0 else (2.0 * intersection) / denom

      macro_dice = float(np.mean(class_dice))
      return (labels, class_dice), macro_dice


def NMI(y_true: np.ndarray, y_pred: np.ndarray,labels: Optional[Sequence]=None) -> float:
      """Compute normalized mutual information between two label arrays. passing labels does nothing.
      
      Note that NMI is best served comparing between different modalities (i.e. CT vs MRI). When modalities are the same, NMI falls short compared to DICE/Jaccard. 
      """
      y_true_arr, y_pred_arr = _validate_1d_arrays(y_true, y_pred)
      return float(metrics.normalized_mutual_info_score(y_true_arr, y_pred_arr))


def PCA(data:np.ndarray, n_components:int=2, visualize_axes:bool=False, whiten: bool=False)-> tuple:
      from sklearn.decomposition import PCA
      import distinctipy
      if n_components>3: visualize_axes=False
      model = PCA(n_components=n_components,whiten=whiten, svd_solver='covariance_eigh')
      model.fit(data)
      data_transformed = model.transform(data)
      
      if visualize_axes:
            if data.ndim != 2 or data.shape[1] != 3:
                  raise ValueError("`visualize_axes=True` requires `data` with shape (n_samples, 3).")
            PC_colors = distinctipy.get_colors(3)
            variance_norm = float(np.linalg.norm(model.explained_variance_))
            if variance_norm == 0.0:
                  variance_norm = 1.0
            data_center = np.mean(data, axis=0)
            data_ranges = np.ptp(data, axis=0)
            half_range = float(np.max(data_ranges) / 2.0)
            if half_range == 0.0:
                  half_range = 1.0
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(data[:,0],data[:,1],data[:,2], alpha=0.2,s=5)
            origin = data_center
            for idx,(component,variance) in enumerate(zip(model.components_, model.explained_variance_)):
                  axis_length = float(variance / variance_norm)
                  direction = component * axis_length
                  ax.quiver(
                        origin[0],
                        origin[1],
                        origin[2],
                        direction[0],
                        direction[1],
                        direction[2],
                        color=PC_colors[idx],
                        arrow_length_ratio=0.1,
                        label=f'PC{idx+1}'
                  )

            if model.components_.shape[0] >= 3:
                  view_normal = model.components_[2]
            elif model.components_.shape[0] == 2:
                  view_normal = np.cross(model.components_[0], model.components_[1])
            else:
                  view_normal = np.array([0.0, 0.0, 1.0])

            view_normal = view_normal / np.linalg.norm(view_normal)
            elev = np.degrees(np.arcsin(view_normal[2]))
            azim = np.degrees(np.arctan2(view_normal[1], view_normal[0]))
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(data_center[0] - half_range, data_center[0] + half_range)
            ax.set_ylim(data_center[1] - half_range, data_center[1] + half_range)
            ax.set_zlim(data_center[2] - half_range, data_center[2] + half_range)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.set_xlabel('R')
            ax.set_ylabel('A')
            ax.set_zlabel('S')
            ax.legend()

            if data_transformed.shape[1] == 3:
                  ax_transformed = fig.add_subplot(122, projection='3d')
                  ax_transformed.scatter(
                        data_transformed[:,0],
                        data_transformed[:,1],
                        data_transformed[:,2],
                        alpha=0.2,
                        s=5
                  )
                  transformed_center = np.mean(data_transformed[:, :3], axis=0)
                  transformed_axes = np.eye(3) * (model.explained_variance_[:3] / variance_norm)[:, np.newaxis]
                  for idx, direction in enumerate(transformed_axes):
                        ax_transformed.quiver(
                              transformed_center[0],
                              transformed_center[1],
                              transformed_center[2],
                              direction[0],
                              direction[1],
                              direction[2],
                              color=PC_colors[idx],
                              arrow_length_ratio=0.1,
                              label=f'PC{idx+1} axis'
                        )
                  ax_transformed.set_xlim(transformed_center[0] - half_range, transformed_center[0] + half_range)
                  ax_transformed.set_ylim(transformed_center[1] - half_range, transformed_center[1] + half_range)
                  ax_transformed.set_zlim(transformed_center[2] - half_range, transformed_center[2] + half_range)
                  ax_transformed.set_box_aspect((1.0, 1.0, 1.0))
                  ax_transformed.set_xlabel('PC1')
                  ax_transformed.set_ylabel('PC2')
                  ax_transformed.set_zlabel('PC3')
            elif data_transformed.shape[1] == 2:
                  ax_transformed = fig.add_subplot(122)
                  ax_transformed.scatter(
                        data_transformed[:,0],
                        data_transformed[:,1],
                        alpha=0.2,
                        s=5
                  )
                  transformed_center = np.mean(data_transformed[:, :2], axis=0)
                  transformed_axes = np.eye(2) * (model.explained_variance_[:2] / variance_norm)[:, np.newaxis]
                  for idx, direction in enumerate(transformed_axes):
                        ax_transformed.quiver(
                              transformed_center[0],
                              transformed_center[1],
                              direction[0],
                              direction[1],
                              angles='xy',
                              scale_units='xy',
                              scale=1.0,
                              color=PC_colors[idx]
                        )
                  ax_transformed.set_xlim(transformed_center[0] - half_range, transformed_center[0] + half_range)
                  ax_transformed.set_ylim(transformed_center[1] - half_range, transformed_center[1] + half_range)
                  ax_transformed.set_aspect('equal', adjustable='box')
                  ax_transformed.set_xlabel('PC1')
                  ax_transformed.set_ylabel('PC2')
            else:
                  ax_transformed = fig.add_subplot(122)
                  ax_transformed.scatter(
                        np.arange(data_transformed.shape[0]),
                        data_transformed[:,0],
                        alpha=0.2,
                        s=5
                  )
                  transformed_center = float(np.mean(data_transformed[:,0]))
                  sample_center = (data_transformed.shape[0] - 1) / 2.0
                  ax_transformed.quiver(
                        sample_center,
                        transformed_center,
                        0.0,
                        float(model.explained_variance_[0] / variance_norm),
                        angles='xy',
                        scale_units='xy',
                        scale=1.0,
                        color=PC_colors[0]
                  )
                  ax_transformed.set_xlim(sample_center - half_range, sample_center + half_range)
                  ax_transformed.set_ylim(transformed_center - half_range, transformed_center + half_range)
                  ax_transformed.set_aspect('equal', adjustable='box')
                  ax_transformed.set_xlabel('Sample')
                  ax_transformed.set_ylabel('PC1')
      return model, data_transformed
