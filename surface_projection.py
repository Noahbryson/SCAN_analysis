from typing import Tuple, List, Optional, Iterable, Self, Hashable
import json
import matplotlib.tri as mtri
import pandas as pd
import subprocess, os, shutil
from pathlib import Path
import numpy as np
import nibabel
from nibabel.cifti2 import cifti2_axes
from nibabel.gifti.gifti import GiftiImage
try: from PyBrain.modules.helper_functions import pdist2,flattenCells,cotangent_laplacian
except ModuleNotFoundError or ImportError: from helper_functions import pdist2 ,flattenCells,cotangent_laplacian # type: ignore
try: from PyBrain.modules.connectivity import networkGraph
except ModuleNotFoundError or ImportError: from connectivity import networkGraph # type: ignore
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pyvista as pv
from heapq import heappush, heappop
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from matplotlib.colors import ListedColormap
import collections

"""Across this API, hemi corresponds to lh/rh, while side corresponds to R/L"""

def annotations_from_gifti_labels(label_map: nibabel.gifti.gifti.GiftiLabelTable):
      """
      annotations_from_gifti_labels pulls labels from GiftiLabelTable out into dictionaries

      Args:
            label_map (nibabel.gifti.gifti.GiftiLabelTable): stored Gifti label table within an .label.gii annotation file

      Returns:
            tuple(dict,dict): returns color map and region map dictionaries. Dictionary keys are the numeric values associated with a given parcel, keys are the same for both dictionaries and should index the .label.gii data array. 
      """
      
      cmap= {}
      region_map={}
      for label in label_map.labels:
            label_name = label.key
            # Color values are 0-255, stored as float in nibabel, so convert to int
            green = int(label.green * 255)
            red = int(label.red * 255)
            blue = int(label.blue * 255)
            alpha = int(label.alpha * 255)
            cmap[label_name] = (red, green, blue, alpha)
            region_map[label_name] = label.label
      return cmap, region_map
      
      
      
def alpha_colormap(color: tuple, N:int=256)->ListedColormap:
      """Create a colormap with fixed RGB and alpha gradient."""
      r, g, b = color

      colors = np.zeros((N, 4))
      colors[:, 0] = r
      colors[:, 1] = g
      colors[:, 2] = b
      colors[:, 3] = np.linspace(0, 1, N)  # alpha gradient

      return ListedColormap(colors)
      

def load_nii_file(fp:Path)->GiftiImage:
      """Load a neuroimaging file with nibabel."""
      return nibabel.load(fp) # type: ignore
def load_and_plot_gifti(fps: List[Path])->pv.Plotter:
      """Load and render GIFTI surfaces in a PyVista plotter."""
      p = pv.Plotter()
      for fp in fps:
            gifti = load_nii_file(fp)
            verts, faces = gifti.agg_data()
            faces = flattenCells(faces)
            mesh = pv.PolyData(verts,faces)
            p.add_mesh(mesh)
      return p
def run_process(cmd:list, cwd=None,env=None,verbose=False):
      """Run a subprocess command with optional logging."""
      if verbose:
            print(">>", " ".join(cmd))
      else:
            print(">>", cmd[0])
      subprocess.run(cmd,check=True,cwd=cwd,env=env)

class Atlas():
      def __init__(self,name,left,right,sourceFolder)->None:
            """  init  ."""
            self.name = name
            self.left_sphere = left
            self.right_sphere = right
            self.source = sourceFolder
            self.pt_sphere_name = 'sphere.reg.surf.gii'
      
      @classmethod
      def fs_LR(cls,atlas_name: str, sourceFolder=Path('/Users/nkb/Documents/HCP-workbench/HCPpipelines/global/templates/standard_mesh_atlases'))-> Self:
            """Fs lr."""
            left = sourceFolder/f'L.sphere.{atlas_name}_fs_LR.surf.gii'
            right = sourceFolder/f'R.sphere.{atlas_name}_fs_LR.surf.gii'
            return cls(atlas_name,left,right,sourceFolder)
      
      @classmethod
      def fs_LR_from_fsav(cls,atlas_name: str, sourceFolder=Path('/Users/nkb/Documents/HCP-workbench/HCPpipelines/global/templates/standard_mesh_atlases'))-> Self:
            """Fs lr from fsav."""
            left = sourceFolder / 'resample_fsaverage' / f"fs_LR-deformed_to-fsaverage.L.sphere.{atlas_name}_fs_LR.surf.gii"
            right= sourceFolder / 'resample_fsaverage' / f"fs_LR-deformed_to-fsaverage.R.sphere.{atlas_name}_fs_LR.surf.gii"
            return cls(atlas_name,left,right,sourceFolder) 

      
      
      
class Project2Subject:
      """
      Project group-space maps back to subject-specific topology.

      Workflow:
      1) Generate subject sphere projected to group sphere topology.
      2) Resample group metric map from group sphere to subject sphere.
      """

      def __init__(self, group_map: str | Path, atlas: Atlas, subject_folder: str | Path, out_root: Optional[str | Path] = None) -> None:
            self.group_map = group_map
            self.atlas = atlas
            self.subject_folder = Path(subject_folder)
            self.out_root = Path(out_root) if out_root is not None else self.subject_folder / "gifti" / "pt-space" / f"group_to_subject.{self.atlas.name}"
            self.out_root.mkdir(parents=True, exist_ok=True)
            self.subject_sphere_projected_dir = self.out_root / "subject_sphere_in_group_topology"
            self.subject_sphere_projected_dir.mkdir(parents=True, exist_ok=True)
            self._cached_cifti_gifti: dict[str, Path] = {}

      def _group_sphere(self, hemi: str) -> Path:
            return self.atlas.left_sphere if hemi.lower() == "lh" else self.atlas.right_sphere

      def _subject_sphere(self, hemi: str) -> Path:
            hemi = hemi.lower()
            candidates = [
                  self.subject_folder / "gifti" / "pt-space" / "surf" / f"{hemi}.{self.atlas.pt_sphere_name}",
                  self.subject_folder / "segmentation" / "surf" / f"{hemi}.sphere.reg.surf.gii",
                  self.subject_folder / "segmentation" / "surf" / f"{hemi[0]}h.sphere.reg",
                  self.subject_folder / "segmentation" / "surf" / f"{hemi[0]}h.sphere.reg.surf.gii",
            ]
            for fp in candidates:
                  if fp.exists():
                        return fp
            raise FileNotFoundError(f"Could not find subject sphere for {hemi}. Checked: {candidates}")

      def _subject_original_sphere(self, hemi: str) -> Path:
            """Resolve subject native/original sphere (non-reg) path for a hemisphere."""
            hemi = hemi.lower()
            candidates = [
                  self.subject_folder / "gifti" / "pt-space" / "surf" / f"{hemi}.sphere.surf.gii",
                  self.subject_folder / "segmentation" / "surf" / f"{hemi}.sphere.surf.gii",
                  self.subject_folder / "segmentation" / "surf" / f"{hemi[0]}h.sphere",
                  self.subject_folder / "segmentation" / "surf" / f"{hemi[0]}h.sphere.surf.gii",
            ]
            for fp in candidates:
                  if fp.exists():
                        return fp
            raise FileNotFoundError(f"Could not find subject original sphere for {hemi}. Checked: {candidates}")

      def _resolve_freesurfer_subject_context(self) -> tuple[str, Path]:
            """
            Resolve FreeSurfer subject id and SUBJECTS_DIR, including nested
            layouts like `<subject_folder>/segmentation/<subject_id>`.
            """
            # subject_folder is directly the FS subject dir
            if (self.subject_folder / "surf").exists():
                  return self.subject_folder.name, self.subject_folder.parent

            seg_root = self.subject_folder / "segmentation"
            if seg_root.exists() and seg_root.is_dir():
                  # segmentation itself is FS subject dir
                  if (seg_root / "surf").exists():
                        return seg_root.name, seg_root.parent
                  # segmentation contains FS subject dir(s)
                  candidates = [p for p in seg_root.iterdir() if p.is_dir() and (p / "surf").exists()]
                  if len(candidates) == 1:
                        fs_subj = candidates[0]
                        return fs_subj.name, fs_subj.parent
                  if len(candidates) > 1:
                        raise ValueError(f"Multiple FreeSurfer subject dirs under {seg_root}: {candidates}")

            # fallback
            return self.subject_folder.name, self.subject_folder.parent

      def _materialize_group_metric(self, hemi: str) -> Path:
            hemi = hemi.lower()
            side = "L" if hemi == "lh" else "R"
            if isinstance(self.group_map, (str, Path)):
                  
                  
                  
                  raw = str(self.group_map)
                  if "[hemi]" in raw:
                        return Path(raw.replace("[hemi]", hemi))
                  if "[side]" in raw:
                        return Path(raw.replace("[side]", side))
                  if ' ' in raw:
                        temp = Path(raw)
                        tempdir = self.out_root / 'temp'
                        tempdir.mkdir(parents=True,exist_ok=True)
                        raw = str(shutil.copy(self.group_map,tempdir/temp.name.replace(' ','_')))
                  fp = Path(raw)
                  if self._is_cifti_file(fp):
                        gifti_paths = self._convert_cifti_to_hemi_gifti(fp)
                        return gifti_paths[hemi]
                  return fp

            if hemi not in self.group_map:
                  raise KeyError(f"group_map dictionary must contain '{hemi}'.")
            val = self.group_map[hemi]
            if isinstance(val, (str, Path)):
                  return Path(val)

            data = np.asarray(val, dtype=np.float32).reshape(-1)
            outfile = self.out_root / f"{self.atlas.name}.{hemi}.group_map.func.gii"
            arr = nibabel.gifti.GiftiDataArray(data=data, intent="NIFTI_INTENT_SHAPE")
            img = nibabel.gifti.GiftiImage(darrays=[arr])
            nibabel.save(img, outfile)
            return outfile

      def _is_cifti_file(self, fp: Path) -> bool:
            """Return True if path points to a CIFTI file."""
            suffixes = "".join(fp.suffixes).lower()
            if suffixes.endswith(".dscalar.nii") or suffixes.endswith(".dtseries.nii") or suffixes.endswith(".dlabel.nii") or suffixes.endswith(".dconn.nii"):
                  return True
            try:
                  img = nibabel.load(fp)
            except Exception:
                  return False
            return isinstance(img, nibabel.cifti2.cifti2.Cifti2Image)

      def _convert_cifti_to_hemi_gifti(self, cifti_fp: Path) -> dict[str, Path]:
            """
            Convert CIFTI cortical grayordinates to hemisphere GIFTI func files.

            Uses the first map for 2D CIFTI data.
            """
            key = str(cifti_fp.resolve())
            if len(self._cached_cifti_gifti) == 2 and all(h in self._cached_cifti_gifti for h in ["lh", "rh"]):
                  return self._cached_cifti_gifti

            img = nibabel.load(cifti_fp)
            if not isinstance(img, nibabel.cifti2.cifti2.Cifti2Image):
                  raise ValueError(f"Expected CIFTI file, got: {cifti_fp}")

            data = np.asarray(img.get_fdata())
            axes = [img.header.get_axis(i) for i in range(data.ndim)]
            bm_dim = None
            bm_axis = None
            for i, ax in enumerate(axes):
                  if isinstance(ax, cifti2_axes.BrainModelAxis):
                        bm_dim = i
                        bm_axis = ax
                        break
            if bm_axis is None or bm_dim is None:
                  raise ValueError(f"No BrainModelAxis found in CIFTI: {cifti_fp}")

            if data.ndim == 1:
                  gray = data
            elif data.ndim == 2:
                  gray = data[:, 0] if bm_dim == 0 else data[0, :]
            else:
                  raise ValueError(f"Unsupported CIFTI dimensionality: {data.ndim}")

            out: dict[str, Path] = {}
            mapping = {
                  "CIFTI_STRUCTURE_CORTEX_LEFT": "lh",
                  "CIFTI_STRUCTURE_CORTEX_RIGHT": "rh",
            }
            for structure_name, slc, model in bm_axis.iter_structures():
                  if structure_name not in mapping:
                        continue
                  if not np.all(model.surface_mask):
                        continue
                  hemi = mapping[structure_name]
                  nvert = model.nvertices[structure_name]
                  arr = np.zeros(nvert, dtype=np.float32)
                  arr[model.vertex] = gray[slc].astype(np.float32)

                  gii_fp = self.out_root / f"{self.atlas.name}.{hemi}.{cifti_fp.stem}.func.gii"
                  gda = nibabel.gifti.GiftiDataArray(data=arr, intent="NIFTI_INTENT_SHAPE")
                  gii = nibabel.gifti.GiftiImage(darrays=[gda])
                  nibabel.save(gii, gii_fp)
                  out[hemi] = gii_fp

            if "lh" not in out or "rh" not in out:
                  raise ValueError(f"CIFTI conversion did not produce both hemispheres for {cifti_fp}")
            self._cached_cifti_gifti = out
            return out

      def project_subject_sphere_to_group_topology(self, process: bool = True) -> dict[str, Path]:
            """Project subject sphere registration to group sphere topology."""
            outputs: dict[str, Path] = {}
            for hemi in ["lh", "rh"]:
                  subj_sphere = self._subject_sphere(hemi)
                  group_sphere = self._group_sphere(hemi)
                  out_fp = self.subject_sphere_projected_dir / f"{hemi}.subject_sphere.projected_to_group_topology.surf.gii"
                  cmd = f"wb_command -surface-resample {subj_sphere} {subj_sphere} {group_sphere} BARYCENTRIC {out_fp}".split()
                  if process:
                        run_process([str(i) for i in cmd], verbose=True)
                  outputs[hemi] = out_fp
            return outputs

      def project_group_map_to_subject(self, process: bool = True, method: str = "BARYCENTRIC") -> dict[str, Path]:
            """Project group map from atlas sphere space to subject sphere space."""
            self.project_subject_sphere_to_group_topology(process=process)
            outputs: dict[str, Path] = {}
            for hemi in ["lh", "rh"]:
                  metric_in = self._materialize_group_metric(hemi)
                  current_sphere = self._group_sphere(hemi)
                  new_sphere = self._subject_sphere(hemi)
                  out_fp = self.out_root / f"{self.atlas.name}.{hemi}.group_map.in_subject_space.func.gii"
                  cmd = f"wb_command -metric-resample {metric_in} {current_sphere} {new_sphere} {method} {out_fp}"
                  tag = "CORTEX_LEFT" if hemi == "lh" else "CORTEX_RIGHT"
                  cmd1 = f"wb_command -set-structure {out_fp} {tag}"
                  if process:
                        run_process([str(i) for i in cmd.split()], verbose=True)
                        run_process([str(i) for i in cmd1.split()], verbose=True)
                  outputs[hemi] = out_fp
            return outputs

      def project_subject_map_to_original_topology(self, subject_map: dict[str, Path], process: bool = True, method: str = "BARYCENTRIC") -> dict[str, Path]:
            """
            Project subject-space map from subject registration sphere to original sphere topology.
            """
            outputs: dict[str, Path] = {}
            for hemi in ["lh", "rh"]:
                  if hemi not in subject_map:
                        continue
                  metric_in = subject_map[hemi]
                  reg_sphere = self._subject_sphere(hemi)
                  orig_sphere = self._subject_original_sphere(hemi)
                  out_root = self.out_root.parent / 'group_to_subject.original'
                  out_root.mkdir(parents=True,exist_ok=True)
                  out_fp = out_root / f"{self.atlas.name}.{hemi}.group_map.in_subject_original_topology.func.gii"
                  cmd = f"wb_command -metric-resample {metric_in} {reg_sphere} {orig_sphere} {method} {out_fp}"
                  tag = "CORTEX_LEFT" if hemi == "lh" else "CORTEX_RIGHT"
                  cmd1 = f"wb_command -set-structure {out_fp} {tag}"
                  if process:
                        run_process([str(i) for i in cmd.split()], verbose=True)
                        run_process([str(i) for i in cmd1.split()], verbose=True)
                  outputs[hemi] = out_fp
            return outputs
      def patient_space_surf2vol(self, surfs: dict[str, Path], vol_path: str | Path, process: bool = True) -> dict[str, Path]:
            """
            Convert original-topology surface maps to subject-aligned volumes using mri_surf2vol.

            Args:
                  surfs: Hemisphere surface-map files to project (`lh`/`rh` in filenames).
                  vol_path: Reference volume defining output grid/alignment.
                  process: Execute FreeSurfer commands if True.

            Returns:
                  Dictionary keyed by hemisphere (`lh`, `rh`) with output volume paths.
            """
            template_vol = Path(vol_path)
            if not template_vol.exists():
                  raise FileNotFoundError(f"Template volume not found: {template_vol}")

            if len(surfs) == 0:
                  subject_space = self.project_group_map_to_subject(process=process, method="BARYCENTRIC")
                  original_maps = self.project_subject_map_to_original_topology(subject_space, process=process, method="BARYCENTRIC")
                  surfs = {h: original_maps[h] for h in ["lh", "rh"] if h in original_maps}

            out_dir = self.out_root.parent / 'volumes'
            out_dir.mkdir(parents=True, exist_ok=True)

            subject_id, subjects_dir = self._resolve_freesurfer_subject_context()
            outputs: dict[str, Path] = {}

            for hemi, surf_fp in surfs.items():
                  surf_fp = Path(surf_fp)
                  if hemi == "":
                        hemi = "lh" if "lh" in surf_fp.name else ("rh" if "rh" in surf_fp.name else "")
                        if hemi == "":
                              raise ValueError(f"Could not infer hemisphere from surface filename: {surf_fp.name}")

                  out_fp = out_dir / f"{surf_fp.stem}.nii.gz"
                  cmd = [
                        "mri_surf2vol",
                        "--surfval", str(surf_fp),
                        "--hemi", hemi,
                        "--template", str(template_vol),
                        "--subject", str(subject_id),
                        "--sd", str(subjects_dir),
                        "--identity", str(subject_id),
                        "--o", str(out_fp),
                  ]
                  if process:
                        run_process(cmd, verbose=True)
                  outputs[hemi] = out_fp

            return outputs


      def run(self, process: bool = True, method: str = "BARYCENTRIC") -> dict[str, Path]:
            """Run full mapping workflow ending in subject original sphere topology."""
            subject_space = self.project_group_map_to_subject(process=process, method=method)
            return self.project_subject_map_to_original_topology(subject_space, process=process, method=method)


class projectAtlas():
      def __init__(self,segmentation_dir:str|Path, atlas:Optional[Atlas]=None, electrode_dir:Optional[str|Path]=None,process_fs: bool=False, process_gifti: bool=True, correct_affine=True,buildFileTree:Optional[Path]=None,distanceThreshold:float=5.0)->None:
            """  init  ."""
            if not isinstance(segmentation_dir,Path):
                  segmentation_dir = Path(segmentation_dir)
            if (not isinstance(electrode_dir,Path) and electrode_dir is not None):
                  electrode_dir = Path(electrode_dir)
            if atlas is None:
                  self.atlas: Atlas = Atlas.fs_LR_from_fsav('32k')
            else: self.atlas: Atlas = atlas
            self.root: Path = segmentation_dir.parent
            self.data_root: Path = self.root/'gifti'
            if buildFileTree is not None:
                  
                  self.build_file_tree_from_BOX(buildFileTree.name,buildFileTree.parent)
            if not os.path.exists(self.data_root):
                  process_fs = True
                  process_gifti = True
            fs_paths = self.fs_seg_to_gifti(seg_path=segmentation_dir,out_path = self.data_root/'pt-space',process = process_fs,correct_affine = correct_affine)
            self.resamp_root,process_gifti = self.surface_resample(fs_paths[0],process_gifti)
            self.metric_resample(fs_paths[1],process_gifti)
            self.label_resample(fs_paths[2],process_gifti)
            self.init_fsLR_template_paths()
            if electrode_dir is not None:
                  try:
                        electrodes = self.load_electrodes_from_dat(electrode_dir)
                        self.surface_map_electrodes(self.resamp_root,electrodes,process=process_gifti,thresh=distanceThreshold)
                        self.electrode_path = self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}' / 'electrodes' / f'[hemi].{self.atlas.name}.electrodes.json'
                  except FileNotFoundError:
                        print(f'no electrodes at path: {electrode_dir}')
            self.lh_plot_surf =  self.midthickness_template_path_fsLR.parent / self.midthickness_template_path_fsLR.name.replace('[hemi]', 'lh')
            self.rh_plot_surf=  self.midthickness_template_path_fsLR.parent / self.midthickness_template_path_fsLR.name.replace('[hemi]', 'rh')
            self.set_default_views()
            
      def init_fsLR_template_paths(self)->None:
            """Init template paths for loading of data files later"""
            
            # fs_LR paths
            self.flatmap_template_path_fsLR = self.atlas.source / f'colin.cerebral.[side].flat.{self.atlas.name}_fs_LR.surf.gii'
            self.sulcal_depth_map_template_path_fsLR = self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}' / 'metric' / f'{self.atlas.name}.[hemi].sulc.func.gii'
            self.annot_file_template_path_fsLR = self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}' / 'label' / f'{self.atlas.name}.[hemi].aparc.a2009s.annot.label.gii'
            self.midthickness_template_path_fsLR = self.data_root / 'pt-space' / f'fs_LR.{self.atlas.name}' / 'surf' / f'{self.atlas.name}.[hemi].midthickness.surf.gii'
            self.additional_ROIs = {}

            # patient specific paths
            #     non-resampled, for naive projection of electrodes into PC space
            self.sulcal_depth_map_template_path_pt = self.data_root /'pt-space'/ 'func' / '[hemi].sulc.func.gii'
            self.annot_file_template_path_pt = self.data_root /'pt-space'/ 'label' / '[hemi].aparc.a2009s.annot.label.gii'
            self.midthickness_template_path_pt = self.data_root / 'pt-space' / 'surf' / '[hemi].midthickness.surf.gii'
      def build_file_tree_from_BOX(self,subject:str,box_path_root:Path)->None:
            """Build file tree from box."""
            if 'UAB' in subject:
                  try:
                        fp = box_path_root / subject
                        if not os.path.exists(self.root/'electrodes'):
                              shutil.copytree(fp/'DataOutput'/f'{subject}_Electrodes',self.root/'electrodes')
                        if not os.path.exists(self.root/'segmentation'):
                              print(f'Building local file tree for {subject} at {self.root}')
                              shutil.copytree(fp/'Generate_Freesurfer_Model'/ 'segmentation'/'surf',self.root/'segmentation'/'surf')
                              shutil.copytree(fp/'Generate_Freesurfer_Model'/ 'segmentation'/'label',self.root/'segmentation'/'label')
                  
                  except FileNotFoundError as e:
                        print(f'Could not build file tree for {subject} due to {e}')
            else:
                  fp = box_path_root / subject / 'IMAGING'
                  try:
                        if not os.path.exists(self.root/'electrodes'):
                              shutil.copytree(fp/'electrodes',self.root/'electrodes')
                        if not os.path.exists(self.root/'segmentation'):
                              print(f'Building local file tree for {subject} at {self.root}')
                              shutil.copytree(fp/'segmentation'/'surf',self.root/'segmentation'/'surf')
                              shutil.copytree(fp/'segmentation'/'label',self.root/'segmentation'/'label')
                        # if not os.path.exists(self.root/'imaging'):
                        #       shutil.copytree(fp/'NIfTI',self.root/'imaging')
                  except FileNotFoundError as e:
                        print(f'Could not build file tree for {subject} due to {e}')
                  
      def get_MRI_mask_ROIs(self)-> List[Path]|None:
            """Get mri mask rois."""
            outfiles = []
            image_dir = self.root/'imaging/images_converted/MR'
            for i in image_dir.glob('*'):
                  for j in i.glob('*masked*'):
                        outfiles.append(j)
            if len(outfiles) == 0: return None
            return outfiles
      
      def project_additional_MRI_ROIs(self,MRI_vols: Optional[List[Path]],process: bool=True)->None:
            # TODO: process MRIs aligned to freesurfer segmentations to surfaces, then warp existing metric files in patient topology to atlas topology
            """Project additional mri rois."""
            if MRI_vols is None: return None
            for i in MRI_vols:
                  name = i.name.replace('.nii','').replace('.nii.gz','')
                  pt_saveDir = self.data_root /'pt-space'/'func'
                  fs_LR_save_dir = self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}/metric'
                  os.makedirs(fs_LR_save_dir,exist_ok=True)
                  surfDir = self.data_root /'pt-space/surf'
                  for j in ['lh','rh']:
                        mid = surfDir / f'{j}.midthickness.surf.gii'
                        pial = surfDir / f'{j}.pial.surf.gii'
                        white = surfDir / f'{j}.white.surf.gii'
                        pt_space_outfile = pt_saveDir / f'{j}.{name}.func.gii'
                        pt_outfile_name = pt_space_outfile.name
                        fs_sphere, hemi, tag = self._checkHemi(pt_outfile_name)
                        cmd = f"wb_command -volume-to-surface-mapping {i} {mid} {pt_space_outfile} -ribbon-constrained {white} {pial}".split()
                        cmd1 = f"wb_command -set-structure {pt_space_outfile} {tag}".split()
                        if process: run_process(cmd); run_process(cmd1)                  

                        fs_lr_hemi = hemi[0].upper()
                        pt_sphere =  surfDir/ f'{hemi}.{self.atlas.pt_sphere_name}'
                        # fs_midthickness = self.atlas.source / f'resample_fsaverage/fs_LR.{fs_lr_hemi}.midthickness_va_avg.{self.atlas.name}_fs_LR.shape.gii'
                        # pt_midthickness = i.parent.parent / f'surf/{hemi}.midthickness.surf.gii'
                        fs_LR_outfile = fs_LR_save_dir / f'{self.atlas.name}.{pt_outfile_name}'
                        # cmd = f"wb_command -metric-resample {i} {pt_sphere} {fs_sphere} ADAP_BARY_AREA {outfile} -area-metrics {pt_midthickness} {fs_midthickness}".split()
                        cmd = f"wb_command -metric-resample {pt_space_outfile} {pt_sphere} {fs_sphere} BARYCENTRIC {fs_LR_outfile}".split()
                        cmd1 = f"wb_command -set-structure {fs_LR_outfile} {tag}".split()
                        if process: run_process(cmd);run_process(cmd1) 

      def _load_freesurfer_label_file(self, label_fp: Path) -> tuple[np.ndarray, np.ndarray]:
            """Load vertex ids and RAS coordinates from a FreeSurfer `.label` file."""
            vertex_ids: list[int] = []
            ras_coords: list[list[float]] = []
            with open(label_fp, "r", encoding="utf-8") as fp:
                  lines = fp.readlines()[2:]
            for line in lines:
                  parts = line.strip().split()
                  if len(parts) < 5:
                        continue
                  vertex_ids.append(int(float(parts[0])))
                  ras_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.asarray(vertex_ids, dtype=int), np.asarray(ras_coords, dtype=float)

      def _split_label_coords_by_hemi(self, ras_coords: np.ndarray) -> dict[str, np.ndarray]:
            """Split label coordinates into hemispheres using RAS x-coordinate sign."""
            if ras_coords.ndim != 2 or ras_coords.shape[1] != 3:
                  raise ValueError("Label coordinates must have shape (n, 3).")

            out: dict[str, np.ndarray] = {}
            lh_mask = ras_coords[:, 0] < 0
            rh_mask = ras_coords[:, 0] > 0
            mid_mask = ~(lh_mask | rh_mask)

            if np.any(lh_mask):
                  out["lh"] = ras_coords[lh_mask]
            if np.any(rh_mask):
                  out["rh"] = ras_coords[rh_mask]

            if np.any(mid_mask):
                  mid_coords = ras_coords[mid_mask]
                  surf_dir = self.data_root / "pt-space" / "surf"
                  lh_verts = np.asarray(load_nii_file(surf_dir / "lh.midthickness.surf.gii").agg_data("pointset"), dtype=float)
                  rh_verts = np.asarray(load_nii_file(surf_dir / "rh.midthickness.surf.gii").agg_data("pointset"), dtype=float)
                  lh_dist, _ = pdist2(mid_coords, lh_verts, num_mins=1)
                  rh_dist, _ = pdist2(mid_coords, rh_verts, num_mins=1)
                  lh_mid = np.asarray(lh_dist).reshape(-1) <= np.asarray(rh_dist).reshape(-1)
                  rh_mid = ~lh_mid
                  if np.any(lh_mid):
                        out["lh"] = np.vstack([out["lh"], mid_coords[lh_mid]]) if "lh" in out else mid_coords[lh_mid]
                  if np.any(rh_mid):
                        out["rh"] = np.vstack([out["rh"], mid_coords[rh_mid]]) if "rh" in out else mid_coords[rh_mid]
            return out

      def project_freesurfer_labels(
            self,
            label_files: str | Path | List[str | Path],
            process: bool = True,
            vertex_value: float = 1.0
      ) -> dict[str, dict[str, dict[str, Path]]]:
            """
            Project FreeSurfer `.label` files to subject surface then resample to fs_LR space.

            The label file coordinates are split into left/right hemispheres in subject RAS
            space, represented as subject-surface metrics in pt-space, then resampled with
            the same sphere-based metric workflow used elsewhere in this module.
            """
            if isinstance(label_files,Path) and os.path.isdir(label_files):
                  fps = [i for i in label_files.glob('*.label')]
            elif isinstance(label_files, (str, Path)):
                  fps = [Path(label_files)]
            else:
                  fps = [Path(fp) for fp in label_files]

            pt_save_dir = self.data_root / "pt-space" / "func"
            fs_lr_save_dir = self.data_root / "pt-space" / f"fs_LR.{self.atlas.name}" / "metric"
            surf_dir = self.data_root / "pt-space" / "surf"
            pt_save_dir.mkdir(parents=True, exist_ok=True)
            fs_lr_save_dir.mkdir(parents=True, exist_ok=True)

            outputs: dict[str, dict[str, Path]] = {}
            for label_fp in fps:
                  name = label_fp.name.replace(".label", "")
                  print(name)
                  _, ras_coords = self._load_freesurfer_label_file(label_fp)
                  if ras_coords.size == 0:
                        raise ValueError(f"No coordinates found in label file: {label_fp}")
                  hemi_coords = self._split_label_coords_by_hemi(ras_coords)
                  if len(hemi_coords) == 0:
                        raise ValueError(f"Could not assign any coordinates to a hemisphere for: {label_fp}")

                  outputs[name] = {}
                  for hemi, hemi_ras in hemi_coords.items():
                        print(hemi)
                        mid_fp = surf_dir / f"{hemi}.midthickness.surf.gii"
                        surf = load_nii_file(mid_fp)
                        surf_verts = np.asarray(surf.agg_data("pointset"), dtype=float)
                        values = np.zeros(surf_verts.shape[0], dtype=np.float32)
                        pt_space_outfile = pt_save_dir / f"{hemi}.{name}.func.gii"
                        if process:
                              print('running pdist')
                              _, label_vert_ids = pdist2(hemi_ras, surf_verts, num_mins=1)
                              print('finished pdist')
                              label_vert_ids = np.unique(np.asarray(label_vert_ids, dtype=int))
                              values[label_vert_ids] = np.float32(vertex_value)
                              arr = nibabel.gifti.GiftiDataArray(data=values, intent="NIFTI_INTENT_SHAPE")
                              img = nibabel.gifti.GiftiImage(darrays=[arr])
                              nibabel.save(img, pt_space_outfile)

                        fs_sphere, _, tag = self._checkHemi(pt_space_outfile.name)
                        cmd = f"wb_command -set-structure {pt_space_outfile} {tag}".split()
                        if process:
                              run_process(cmd)

                        pt_sphere = surf_dir / f"{hemi}.{self.atlas.pt_sphere_name}"
                        fs_lr_outfile = fs_lr_save_dir / f"{self.atlas.name}.{pt_space_outfile.name}"
                        cmd = f"wb_command -metric-resample {pt_space_outfile} {pt_sphere} {fs_sphere} BARYCENTRIC {fs_lr_outfile}".split()
                        cmd1 = f"wb_command -set-structure {fs_lr_outfile} {tag}".split()
                        if process:
                              run_process(cmd)
                              run_process(cmd1)

                        outputs[name][hemi] = {"pt": pt_space_outfile, "fs_LR": fs_lr_outfile}
            return outputs

      def project_bipolar_electrodes(self,electrode_dir,thresh:float=5.0)->None:
                  """Project bipolar electrodes."""
                  try:
                        electrodes = self.load_electrodes_from_dat(electrode_dir)
                        electrodes = self.__make_bipolar_electrodes(electrodes)
                        self.surface_map_electrodes(self.resamp_root,electrodes,bipolar=True,process=True, thresh=thresh)
                  except FileNotFoundError:
                        print(f'no electrodes at path: {electrode_dir}')

      def __make_bipolar_electrodes(self,electrodes: dict)-> dict:
            """  make bipolar electrodes."""
            import re
            bipolar_electrodes = {}
            trajectories = sorted(set(['_'.join(i.split('_')[0:-1]) for i in electrodes]))
            for t in trajectories:
                  electrode_indices = {int(i.split('_')[-1]):electrodes[i] for i in electrodes if t in i}
                  sorted_indices = sorted(electrode_indices)
                  for idx,i in enumerate(sorted_indices[0:-1]):
                        if sorted_indices[idx+1] - i == 1:
                              chan_label = f'{i}-b-{sorted_indices[idx+1]}'
                              c1, c2 = electrode_indices[i], electrode_indices[sorted_indices[idx+1]]
                              location = np.mean(np.vstack((c1,c2)),axis=0)
                              bipolar_electrodes['_'.join([t,chan_label])] = location
                        else: continue
            return bipolar_electrodes

      def load_cifti_data(self,fp,name)->None:
            """Load cifti data."""
            def extract_cifti_surface_maps(cifti_img, map_index=0, fill=np.nan)-> tuple:
                  
                  """Extract cifti surface maps."""
                  data = cifti_img.get_fdata()
                  axes = [cifti_img.header.get_axis(i) for i in range(data.ndim)]
            
                  bm_dim = None
                  bm_axis = None
                  for i, ax in enumerate(axes):
                        if isinstance(ax, cifti2_axes.BrainModelAxis):
                              bm_dim = i
                              bm_axis = ax
                              break

                  if bm_axis is None:
                        raise ValueError("No BrainModelAxis found in this CIFTI file.")

                  # pull out a single grayordinate vector
                  if data.ndim == 1:
                        gray = data
                  elif data.ndim == 2:
                        if bm_dim == 0:
                              gray = data[:, map_index]
                        elif bm_dim == 1:
                              gray = data[map_index, :]
                        else:
                              raise ValueError("Unsupported 2D layout.")
                  else:
                        raise ValueError(f"Unsupported CIFTI dimensionality: {data.ndim}")

                  out = {}

                  for structure_name, slc, model in bm_axis.iter_structures():
                        if not np.all(model.surface_mask):
                        # skip mixed/volume-only structures for flatmapping
                              continue

                        nvert = model.nvertices[structure_name]
                        arr = np.full(nvert, fill, dtype=gray.dtype)
                        arr[model.vertex] = gray[slc]
                        out[structure_name] = arr

                  return out, cifti_img, bm_axis
            cifti_img = nibabel.load(fp)
            axis0 = cifti_img.header.get_axis(0)
            axis1 = cifti_img.header.get_axis(1)
            bm_axis = axis0 if isinstance(axis0, cifti2_axes.BrainModelAxis) else axis1
            other_axis = axis1 if bm_axis is axis0 else axis0
            for i in range(len(other_axis)):
                  out, img, bm_axis = extract_cifti_surface_maps(cifti_img,map_index=i)
                  mapping = {'CIFTI_STRUCTURE_CORTEX_LEFT':'lh','CIFTI_STRUCTURE_CORTEX_RIGHT':'rh'}
                  res = {mapping[i]:j for i,j in out.items()}
                  if len(other_axis) > 1:
                        key = f'{name}_axis{i}'
                  else: key = name
                  self.additional_ROIs[key] = res
      def load_cifti_as_ROI(self, fp: str | Path, name: Hashable, map_index: int=0, threshold: float=0.0) -> None:
            """Load cortical data from a CIFTI file as an ROI overlay and ignore volumetric components."""
            def project_volume_to_surface(volume_fp: Path, roi_name: Hashable) -> dict[str, Path]:
                  pt_save_dir = self.data_root / 'pt-space' / 'func'
                  fs_lr_save_dir = self.data_root / 'pt-space' / f'fs_LR.{self.atlas.name}' / 'metric'
                  surf_dir = self.data_root / 'pt-space' / 'surf'
                  pt_save_dir.mkdir(parents=True, exist_ok=True)
                  fs_lr_save_dir.mkdir(parents=True, exist_ok=True)

                  out: dict[str, Path] = {}
                  for hemi in ['lh', 'rh']:
                        mid = surf_dir / f'{hemi}.midthickness.surf.gii'
                        pial = surf_dir / f'{hemi}.pial.surf.gii'
                        white = surf_dir / f'{hemi}.white.surf.gii'
                        pt_space_outfile = pt_save_dir / f'{hemi}.{roi_name}.func.gii'
                        fs_sphere, _, tag = self._checkHemi(pt_space_outfile.name)
                        cmd = f"wb_command -volume-to-surface-mapping {volume_fp} {mid} {pt_space_outfile} -ribbon-constrained {white} {pial}".split()
                        cmd1 = f"wb_command -set-structure {pt_space_outfile} {tag}".split()
                        run_process(cmd)
                        run_process(cmd1)

                        pt_sphere = surf_dir / f'{hemi}.{self.atlas.pt_sphere_name}'
                        fs_lr_outfile = fs_lr_save_dir / f'{self.atlas.name}.{pt_space_outfile.name}'
                        cmd = f"wb_command -metric-resample {pt_space_outfile} {pt_sphere} {fs_sphere} BARYCENTRIC {fs_lr_outfile}".split()
                        cmd1 = f"wb_command -set-structure {fs_lr_outfile} {tag}".split()
                        run_process(cmd)
                        run_process(cmd1)
                        out[hemi] = fs_lr_outfile
                  return out

            def extract_cortical_surface_maps(
                  cifti_img: nibabel.cifti2.cifti2.Cifti2Image,
                  map_index: int=0,
                  fill: float=0.0
            ) -> dict[str, np.ndarray]:
                  data = np.asarray(cifti_img.get_fdata())
                  axes = [cifti_img.header.get_axis(i) for i in range(data.ndim)]

                  bm_dim: Optional[int] = None
                  bm_axis: Optional[cifti2_axes.BrainModelAxis] = None
                  for idx, axis in enumerate(axes):
                        if isinstance(axis, cifti2_axes.BrainModelAxis):
                              bm_dim = idx
                              bm_axis = axis
                              break

                  if bm_axis is None or bm_dim is None:
                        raise ValueError("No BrainModelAxis found in this CIFTI file.")

                  if data.ndim == 1:
                        gray = data
                  elif data.ndim == 2:
                        other_dim = 1 - bm_dim
                        other_axis = axes[other_dim]
                        if map_index < 0 or map_index >= len(other_axis):
                              raise IndexError(f"map_index {map_index} out of range for axis length {len(other_axis)}.")
                        gray = data[:, map_index] if bm_dim == 0 else data[map_index, :]
                  else:
                        raise ValueError(f"Unsupported CIFTI dimensionality: {data.ndim}")

                  mapping = {
                        "CIFTI_STRUCTURE_CORTEX_LEFT": "lh",
                        "CIFTI_STRUCTURE_CORTEX_RIGHT": "rh",
                  }
                  out: dict[str, np.ndarray] = {}
                  for structure_name, slc, model in bm_axis.iter_structures():
                        if structure_name not in mapping:
                              continue
                        hemi = mapping[structure_name]
                        nvert = int(model.nvertices[structure_name])
                        arr = np.full(nvert, fill, dtype=np.float32)
                        arr[np.asarray(model.vertex, dtype=int)] = np.asarray(gray[slc], dtype=np.float32)
                        out[hemi] = arr
                  return out

            fp = Path(fp)
            hemisphere_values: dict[str, np.ndarray]
            img = nibabel.load(fp)
            if isinstance(img, nibabel.cifti2.cifti2.Cifti2Image):
                  hemisphere_values = extract_cortical_surface_maps(img, map_index=map_index, fill=0.0)
            else:
                  projected = project_volume_to_surface(fp, name)
                  hemisphere_values = {}
                  for hemi, hemi_fp in projected.items():
                        hemisphere_values[hemi] = np.asarray(load_nii_file(hemi_fp).agg_data(), dtype=np.float32).reshape(-1)

            unique_vals: set[float | int] = set()
            for hemi, data in hemisphere_values.items():
                  roi_data = np.asarray(data, dtype=np.float32)
                  roi_data = np.where(roi_data > threshold, roi_data, 0.0)
                  hemisphere_values[hemi] = roi_data
                  unique_vals.update(np.unique(roi_data).tolist())

            sorted_vals = sorted(unique_vals)
            value_map: dict[float | int, str] = {0.0: "non-indexed"}
            color_map: dict[str, tuple[float, float, float, float]] = {"non-indexed": (0.1, 0.1, 0.1, 0.0)}

            nonzero_vals = [value for value in sorted_vals if float(value) != 0.0]
            if len(nonzero_vals) == 1:
                  value_map[nonzero_vals[0]] = str(name)
                  color_map[str(name)] = (1.0, 0.0, 0.0, 1.0)
            elif len(nonzero_vals) > 1:
                  import distinctipy
                  gen_colors = distinctipy.get_colors(len(nonzero_vals))
                  for idx, value in enumerate(nonzero_vals):
                        label_name = f"{name}_{value}"
                        value_map[value] = label_name
                        color_map[label_name] = tuple(gen_colors[idx]) + (1.0,)

            self.add_additional_ROI_value_map(name, hemisphere_values, value_map, color_map)
      def load_binary_gifti_as_ROI(self, fp: str | Path | List[str | Path] | dict[str, str | Path], name: Hashable) -> None:
            """Load hemisphere `.func.gii` ROI data into `additional_ROIs`."""
            def infer_hemi(path: Path) -> Optional[str]:
                  tokens = path.name.lower().split(".")
                  for token in tokens:
                        if token in {"lh", "rh"}:
                              return token
                        if token == "l":
                              return "lh"
                        if token == "r":
                              return "rh"
                  return None

            def counterpart_path(path: Path, hemi: str) -> Optional[Path]:
                  replacements = [
                        (f"{hemi}.", f"{'rh' if hemi == 'lh' else 'lh'}."),
                        (f".{hemi}.", f".{'rh' if hemi == 'lh' else 'lh'}."),
                        (f"{hemi}_", f"{'rh' if hemi == 'lh' else 'lh'}_"),
                        (f"_{hemi}", f"_{'rh' if hemi == 'lh' else 'lh'}"),
                        (f".{'l' if hemi == 'lh' else 'r'}.", f".{'r' if hemi == 'lh' else 'l'}."),
                        (f"_{'l' if hemi == 'lh' else 'r'}_", f"_{'r' if hemi == 'lh' else 'l'}_"),
                  ]
                  for old, new in replacements:
                        candidate = Path(str(path).replace(old, new, 1))
                        if candidate != path and candidate.exists():
                              return candidate
                  return None

            def normalize_inputs(raw: str | Path | List[str | Path] | dict[str, str | Path]) -> dict[str, Path]:
                  if isinstance(raw, dict):
                        out: dict[str, Path] = {}
                        for hemi, value in raw.items():
                              hemi_key = hemi.lower()
                              if hemi_key not in {"lh", "rh"}:
                                    raise ValueError("GIFTI ROI dict keys must be 'lh' or 'rh'.")
                              out[hemi_key] = Path(value)
                        return out

                  if isinstance(raw, (str, Path)):
                        fps = [Path(raw)]
                  else:
                        fps = [Path(i) for i in raw]

                  if len(fps) == 0:
                        raise ValueError("At least one GIFTI file path is required.")

                  out = {}
                  for path in fps:
                        hemi = infer_hemi(path)
                        if hemi is None:
                              raise ValueError(f"Could not infer hemisphere from filename: {path}")
                        out[hemi] = path

                  if len(out) == 1:
                        hemi = next(iter(out))
                        candidate = counterpart_path(out[hemi], hemi)
                        if candidate is not None:
                              out["rh" if hemi == "lh" else "lh"] = candidate

                  return out

            hemi_paths = normalize_inputs(fp)
            hemisphere_values: dict[str, np.ndarray] = {}
            unique_vals: set[float | int] = set()

            for hemi, path in hemi_paths.items():
                  if not path.exists():
                        raise FileNotFoundError(path)
                  data = np.asarray(load_nii_file(path).agg_data(), dtype=np.float32).reshape(-1)
                  data[data>0]=1
                  data = np.asarray(data,dtype=np.uint8)
                  hemisphere_values[hemi] = data
                  unique_vals.update(np.unique(data).tolist())

            sorted_vals = sorted(unique_vals)
            value_map: dict[float | int, str] = {}
            color_map: dict[str, tuple[float, float, float, float]] = {}
            import distinctipy
            gen_colors = distinctipy.get_colors(len(sorted_vals))
            for c,value in enumerate(sorted_vals):
                  if float(value) == 0.0:
                        value_map[value] = "non-indexed"
                        color_map["non-indexed"] = (0.1, 0.1, 0.1, 0.0)
                  else:
                        value_map[value] = name
                        color_map[name] = gen_colors[c]

            self.add_additional_ROI_value_map(name, hemisphere_values, value_map, color_map if len(color_map) > 0 else None)
      def load_gifti_label_as_ROI(self, fp: str | Path | List[str | Path] | dict[str, str | Path], name: Hashable) -> None:
            """Load hemisphere `.label.gii` ROI data into `additional_ROIs`."""
            def infer_hemi(path: Path) -> Optional[str]:
                  tokens = path.name.lower().split(".")
                  for token in tokens:
                        if token in {"lh", "rh"}:
                              return token
                        if token == "l":
                              return "lh"
                        if token == "r":
                              return "rh"
                  return None

            def counterpart_path(path: Path, hemi: str) -> Optional[Path]:
                  replacements = [
                        (f"{hemi}.", f"{'rh' if hemi == 'lh' else 'lh'}."),
                        (f".{hemi}.", f".{'rh' if hemi == 'lh' else 'lh'}."),
                        (f"{hemi}_", f"{'rh' if hemi == 'lh' else 'lh'}_"),
                        (f"_{hemi}", f"_{'rh' if hemi == 'lh' else 'lh'}"),
                        (f".{'l' if hemi == 'lh' else 'r'}.", f".{'r' if hemi == 'lh' else 'l'}."),
                        (f"_{'l' if hemi == 'lh' else 'r'}_", f"_{'r' if hemi == 'lh' else 'l'}_"),
                  ]
                  for old, new in replacements:
                        candidate = Path(str(path).replace(old, new, 1))
                        if candidate != path and candidate.exists():
                              return candidate
                  return None

            def normalize_inputs(raw: str | Path | List[str | Path] | dict[str, str | Path]) -> dict[str, Path]:
                  if isinstance(raw, dict):
                        out: dict[str, Path] = {}
                        for hemi, value in raw.items():
                              hemi_key = hemi.lower()
                              if hemi_key not in {"lh", "rh"}:
                                    raise ValueError("GIFTI ROI dict keys must be 'lh' or 'rh'.")
                              out[hemi_key] = Path(value)
                        return out

                  if isinstance(raw, (str, Path)):
                        fps = [Path(raw)]
                  else:
                        fps = [Path(i) for i in raw]

                  if len(fps) == 0:
                        raise ValueError("At least one GIFTI file path is required.")

                  out: dict[str, Path] = {}
                  for path in fps:
                        hemi = infer_hemi(path)
                        if hemi is None:
                              raise ValueError(f"Could not infer hemisphere from filename: {path}")
                        out[hemi] = path

                  if len(out) == 1:
                        hemi = next(iter(out))
                        candidate = counterpart_path(out[hemi], hemi)
                        if candidate is not None:
                              out["rh" if hemi == "lh" else "lh"] = candidate

                  return out

            hemi_paths = normalize_inputs(fp)
            hemisphere_values: dict[str, np.ndarray] = {}
            value_map: dict[int, str] = {}
            color_map: dict[str, tuple[float, float, float, float]] = {}

            for hemi, path in hemi_paths.items():
                  if not path.exists():
                        raise FileNotFoundError(path)
                  img = load_nii_file(path)
                  data = np.asarray(img.agg_data(), dtype=np.int32).reshape(-1)
                  hemisphere_values[hemi] = data
                  for label in img.labeltable.labels:
                        label_key = int(label.key)
                        label_name = str(label.label)
                        value_map[label_key] = label_name
                        color_map[label_name] = (
                              float(label.red),
                              float(label.green),
                              float(label.blue),
                              float(label.alpha)
                        )

            if 0 not in value_map:
                  value_map[0] = "non-indexed"
            if "non-indexed" not in color_map:
                  color_map["non-indexed"] = (0.1, 0.1, 0.1, 0.0)

            self.add_additional_ROI_value_map(name, hemisphere_values, value_map, color_map)
      def update_additional_ROI_value_map(self,
            key:Hashable,
            value_map:dict,
            color_map:Optional[dict]=None
            )->None:
            """Update additional roi value map."""
            self.additional_ROIs[key]['vmap'] = value_map
            if color_map is None:
                  color_map = self.make_cmap_from_vmap(value_map)
            if 'non-indexed' in color_map:
                  color_map['non-indexed'] = (0.1,0.1,0.1,0)
            self.additional_ROIs[key]['cmap'] = color_map
      def update_ROI_cmap(self,key:Hashable,cmap:dict[Hashable,tuple])->None:
            """update an ROI cmap with an additional dictionary. Replace existing keys with new values and add keys"""
            temp:dict = self.additional_ROIs[key]['cmap']
            temp.update(cmap)
            self.additional_ROIs[key]['cmap'] = temp
            
            
      def make_cmap_from_vmap(self,value_map)-> dict:
            """Make cmap from vmap."""
            import distinctipy
            colors = distinctipy.get_colors(len(value_map))
            return {k:v for k,v in zip(value_map.values(),colors)}
      
      def add_additional_ROI_value_map(self,
            key: Hashable,
            hemisphere_values: dict,
            value_map: dict,
            color_map: Optional[dict]=None,
            overwrite: bool=False) -> None:
            """
            add_additional_ROI_value_map update value maps with lh/rh value maps analogous to a gifti output, a value mapping and a color mapping (optional) for visualization and analysis

            Args:
                  key (Hashable): Key to assign to the ROI map
                  hemisphere_values (dict): dictionary with keys lh and rh containing values for each vertex of a given surface
                  value_map (dict): links float values in hemisphere values with specific labels
                  color_map (Optional[dict], optional): link between the labels mapped in value_map with specific colors. Color mapping will be randomly generated from distinct colors if no map is passed. Defaults to None.
                  overwrite (bool, optional): fully replace an existing ROI entry instead of merging hemisphere data. Defaults to False.
            """
            hemi_keys = [hemi for hemi in ['lh', 'rh'] if hemi in hemisphere_values]
            if len(hemi_keys) == 0:
                  raise ValueError("hemisphere_values must contain at least one of 'lh' or 'rh'.")

            if overwrite or key not in self.additional_ROIs or not isinstance(self.additional_ROIs[key], dict):
                  self.additional_ROIs[key] = dict(hemisphere_values)
            else:
                  existing = self.additional_ROIs[key]
                  if len(hemi_keys) == 1:
                        hemi = hemi_keys[0]
                        existing[hemi] = hemisphere_values[hemi]
                  else:
                        for hemi in hemi_keys:
                              existing[hemi] = hemisphere_values[hemi]
                  self.additional_ROIs[key] = existing
            self.update_additional_ROI_value_map(key,value_map=value_map,color_map=color_map)
      
      def return_ROI_mapping(self,key: Hashable)-> dict:
            vmap_main = self.additional_ROIs[key]
            out = {}
            def remap_array(arr: np.ndarray, mapping: dict[Hashable, Hashable]) -> np.ndarray:
                  out: np.ndarray = np.empty(arr.shape,dtype=object)

                  # Fast path: integer array + integer keys
                  if np.issubdtype(arr.dtype, np.integer) and all(isinstance(k, (int, np.integer)) for k in mapping.values()):
                        keys: np.ndarray = np.array(sorted(mapping.keys()))
                        vals: np.ndarray = np.array([mapping[k] for k in keys], dtype=out.dtype)
                        idx: np.ndarray = np.searchsorted(keys, out)
                        hit: np.ndarray = (idx < keys.size) & (keys[idx] == out)
                        out[hit] = vals[idx[hit]]
                        out = np.asarray(out,dtype=int)
                        return out
                  # Generic path: works for strings/object/hashables
                  for idx,i in enumerate(arr):
                        try:
                              out[idx] = mapping[i]
                        except KeyError:
                              out[idx] = 'na'
                        

                  return out

            for i in ['lh','rh']:
                  out[i]= remap_array(vmap_main[i],vmap_main['vmap'])
            return out
                        
                  
            

      def add_annotation_as_additional_ROI(self, key: Hashable='annotations')->None:
            """
            Register cortex annotation labels as an ROI-like entry in `additional_ROIs`.
            """
            out = {}
            value_map = {}
            color_map = {}
            for hemi in ['lh', 'rh']:
                  side = self._hemi2side(hemi)
                  _, _, annot_file = self._load_surface_maps(side)
                  annot_data = annot_file.agg_data()
                  out[hemi] = annot_data
                  for label in annot_file.labeltable.labels:
                        value_map[int(label.key)] = str(label.label)
                        color_map[str(label.label)] = (
                              float(label.red),
                              float(label.green),
                              float(label.blue),
                              float(label.alpha)
                        )
            self.add_additional_ROI_value_map(key, out, value_map, color_map)

      def split_connected_surface_roi(
            self,
            key: Hashable,
            target_value: Optional[Hashable]=None,
            min_component_size: int=1,
            overwrite: bool=False
      ) -> list[Hashable]:
            """Split a masked ROI into disconnected surface components and register them as subROIs."""
            def sanitize_token(value: Hashable) -> str:
                  token = str(value)
                  return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in token)

            def resolve_target_values(roi_entry: dict) -> tuple[list[Hashable], str, str]:
                  vmap = roi_entry['vmap']
                  nonzero_values = sorted([value for value in vmap.keys() if value != 0])
                  if target_value is None:
                        if len(nonzero_values) == 0:
                              raise ValueError(f"ROI '{key}' contains no non-zero values.")
                        if len(nonzero_values) > 1:
                              raise ValueError(f"ROI '{key}' is multi-class. Pass target_value to select one label.")
                        selected = [nonzero_values[0]]
                  else:
                        selected = [value for value, label in vmap.items() if value == target_value or label == target_value]
                        if len(selected) == 0:
                              raise KeyError(f"target_value '{target_value}' not found in ROI '{key}'.")
                  label_name = str(vmap[selected[0]])
                  return selected, label_name, sanitize_token(label_name)

            def connected_components(mask: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
                  active_vertices = np.flatnonzero(mask)
                  if active_vertices.size == 0:
                        return []
                  active_mask = np.zeros(mask.shape[0], dtype=bool)
                  active_mask[active_vertices] = True
                  adjacency: dict[int, set[int]] = {int(vertex): set() for vertex in active_vertices}
                  roi_faces = faces[np.all(active_mask[faces], axis=1)]
                  for tri in roi_faces:
                        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
                        adjacency[a].update((b, c))
                        adjacency[b].update((a, c))
                        adjacency[c].update((a, b))

                  visited: set[int] = set()
                  out: list[np.ndarray] = []
                  for start in active_vertices:
                        start_idx = int(start)
                        if start_idx in visited:
                              continue
                        stack = [start_idx]
                        component: list[int] = []
                        visited.add(start_idx)
                        while len(stack) > 0:
                              node = stack.pop()
                              component.append(node)
                              for neighbor in adjacency[node]:
                                    if neighbor not in visited:
                                          visited.add(neighbor)
                                          stack.append(neighbor)
                        out.append(np.asarray(component, dtype=int))
                  out.sort(key=len, reverse=True)
                  return out

            if key not in self.additional_ROIs:
                  raise KeyError(f"ROI '{key}' not found in additional_ROIs.")
            roi_entry = self.additional_ROIs[key]
            if not isinstance(roi_entry, dict):
                  raise ValueError(f"ROI '{key}' is not stored as a hemisphere-valued dictionary.")

            selected_values, label_name, label_token = resolve_target_values(roi_entry)
            source_color = roi_entry['cmap'].get(label_name, (1.0, 0.0, 0.0, 1.0))
            value_map = {0: 'non-indexed', 1: label_name}
            color_map = {'non-indexed': (0.1, 0.1, 0.1, 0.0), label_name: source_color}
            subkeys: list[Hashable] = []

            for hemi in ['lh', 'rh']:
                  if hemi not in roi_entry:
                        continue
                  data = np.asarray(roi_entry[hemi])
                  mask = np.isin(data, selected_values)
                  if not np.any(mask):
                        continue

                  surf_fp = self.midthickness_template_path_fsLR.parent / self.midthickness_template_path_fsLR.name.replace('[hemi]', hemi)
                  faces = np.asarray(load_nii_file(surf_fp).agg_data('triangle'), dtype=int)
                  components = [component for component in connected_components(mask, faces) if len(component) >= min_component_size]
                  if len(components) == 0:
                        continue

                  for idx, component in enumerate(components):
                        subkey = f"{key}_{label_token}_{hemi}_{idx}"
                        subroi = np.zeros(data.shape[0], dtype=np.uint8)
                        subroi[component] = 1
                        self.add_additional_ROI_value_map(
                              key=subkey,
                              hemisphere_values={hemi: subroi},
                              value_map=value_map,
                              color_map=color_map,
                              overwrite=overwrite
                        )
                        subkeys.append(subkey)

            return subkeys

      def refine_split_surface_roi(
            self,
            source_key: Hashable,
            subroi_keys: list[Hashable],
            target_value: Optional[Hashable]=None,
            representative_vertex_method: str='centroid_vertex',
            max_rois: Optional[int]=None,
            overwrite: bool=True
      ) -> list[Hashable]:
            """Refine split surface ROIs by assigning the source ROI vertices to the nearest subROI representative."""
            import scipy.sparse as sp
            from scipy.sparse.csgraph import dijkstra

            def resolve_target_values(roi_entry: dict) -> list[Hashable]:
                  vmap = roi_entry['vmap']
                  nonzero_values = sorted([value for value in vmap.keys() if value != 0])
                  if target_value is None:
                        if len(nonzero_values) == 0:
                              raise ValueError(f"ROI '{source_key}' contains no non-zero values.")
                        if len(nonzero_values) > 1:
                              raise ValueError(f"ROI '{source_key}' is multi-class. Pass target_value to select one label.")
                        return [nonzero_values[0]]
                  selected = [value for value, label in vmap.items() if value == target_value or label == target_value]
                  if len(selected) == 0:
                        raise KeyError(f"target_value '{target_value}' not found in ROI '{source_key}'.")
                  return selected

            def build_surface_graph(verts: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
                  edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(int)
                  edges = np.sort(edges, axis=1)
                  edges = np.unique(edges, axis=0)
                  weights = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
                  row = np.concatenate([edges[:, 0], edges[:, 1]])
                  col = np.concatenate([edges[:, 1], edges[:, 0]])
                  data = np.concatenate([weights, weights])
                  return sp.csr_matrix((data, (row, col)), shape=(verts.shape[0], verts.shape[0]))

            def representative_vertex(component: np.ndarray, verts: np.ndarray) -> int:
                  if representative_vertex_method != 'centroid_vertex':
                        raise ValueError(f"Unsupported representative_vertex_method '{representative_vertex_method}'.")
                  component_verts = verts[component]
                  centroid = np.mean(component_verts, axis=0)
                  local_idx = int(np.argmin(np.linalg.norm(component_verts - centroid, axis=1)))
                  return int(component[local_idx])

            if source_key not in self.additional_ROIs:
                  raise KeyError(f"ROI '{source_key}' not found in additional_ROIs.")
            source_entry = self.additional_ROIs[source_key]
            if not isinstance(source_entry, dict):
                  raise ValueError(f"ROI '{source_key}' is not stored as a hemisphere-valued dictionary.")
            if len(subroi_keys) == 0:
                  return []
            if max_rois is not None and max_rois < 1:
                  raise ValueError("max_rois must be >= 1 when provided.")

            selected_values = resolve_target_values(source_entry)
            refined_keys: list[Hashable] = []
            for hemi in ['lh', 'rh']:
                  if hemi not in source_entry:
                        continue

                  hemi_subkeys = [subkey for subkey in subroi_keys if subkey in self.additional_ROIs and hemi in self.additional_ROIs[subkey]]
                  if len(hemi_subkeys) == 0:
                        continue

                  source_data = np.asarray(source_entry[hemi])
                  source_mask = np.isin(source_data, selected_values)
                  source_indices = np.flatnonzero(source_mask)
                  if source_indices.size == 0:
                        continue

                  surf_fp = self.midthickness_template_path_fsLR.parent / self.midthickness_template_path_fsLR.name.replace('[hemi]', hemi)
                  surf = load_nii_file(surf_fp)
                  verts = np.asarray(surf.agg_data('pointset'), dtype=float)
                  faces = np.asarray(surf.agg_data('triangle'), dtype=int)
                  graph = build_surface_graph(verts, faces)

                  component_records: list[tuple[Hashable, np.ndarray]] = []
                  for subkey in hemi_subkeys:
                        subroi_data = np.asarray(self.additional_ROIs[subkey][hemi])
                        component = np.flatnonzero(subroi_data > 0)
                        if component.size == 0:
                              continue
                        component_records.append((subkey, component))

                  if len(component_records) == 0:
                        continue

                  component_records.sort(key=lambda item: item[1].size, reverse=True)
                  if max_rois is not None and len(component_records) > max_rois:
                        component_records = component_records[:max_rois]

                  kept_subkeys = [subkey for subkey, _ in component_records]
                  representatives = np.asarray(
                        [representative_vertex(component, verts) for _, component in component_records],
                        dtype=int
                  )
                  dist = dijkstra(graph, directed=False, indices=representatives)
                  nearest_component = np.argmin(dist[:, source_indices], axis=0)
                  for comp_idx, subkey in enumerate(kept_subkeys):
                        assigned = source_indices[nearest_component == comp_idx]
                        refined = np.zeros(source_data.shape[0], dtype=np.uint8)
                        refined[assigned] = 1
                        roi_entry = self.additional_ROIs[subkey]
                        self.add_additional_ROI_value_map(
                              key=subkey,
                              hemisphere_values={hemi: refined},
                              value_map=roi_entry['vmap'],
                              color_map=roi_entry['cmap'],
                              overwrite=overwrite
                        )
                        refined_keys.append(subkey)

                  dropped_subkeys = [subkey for subkey in hemi_subkeys if subkey not in kept_subkeys]
                  for subkey in dropped_subkeys:
                        roi_entry = self.additional_ROIs[subkey]
                        empty = np.zeros(source_data.shape[0], dtype=np.uint8)
                        self.add_additional_ROI_value_map(
                              key=subkey,
                              hemisphere_values={hemi: empty},
                              value_map=roi_entry['vmap'],
                              color_map=roi_entry['cmap'],
                              overwrite=overwrite
                        )

            return refined_keys

      def merge_subroi_keys_to_multilabel(
            self,
            subroi_keys: list[Hashable],
            merged_key: Hashable,
            overwrite: bool=False
      ) -> Hashable:
            """Merge binary subROIs into one multilabel ROI with a unique integer code per subROI."""
            if len(subroi_keys) == 0:
                  raise ValueError("subroi_keys must contain at least one ROI key.")

            hemisphere_values: dict[str, np.ndarray] = {}
            value_map: dict[int, str] = {0: 'non-indexed'}
            color_map: dict[str, tuple[float, float, float, float]] = {'non-indexed': (0.1, 0.1, 0.1, 0.0)}

            for code, subkey in enumerate(subroi_keys, start=1):
                  if subkey not in self.additional_ROIs:
                        raise KeyError(f"SubROI '{subkey}' not found in additional_ROIs.")
                  roi_entry = self.additional_ROIs[subkey]
                  if not isinstance(roi_entry, dict):
                        raise ValueError(f"SubROI '{subkey}' is not stored as a hemisphere-valued dictionary.")

                  value_map[code] = str(subkey)

                  roi_vmap = roi_entry.get('vmap', {})
                  roi_cmap = roi_entry.get('cmap', {})
                  positive_labels = [roi_vmap[value] for value in roi_vmap if value != 0]
                  if len(positive_labels) > 0 and positive_labels[0] in roi_cmap:
                        color = roi_cmap[positive_labels[0]]
                  else:
                        color = (1.0, 0.0, 0.0, 1.0)
                  color_map[str(subkey)] = color

                  for hemi in ['lh', 'rh']:
                        if hemi not in roi_entry:
                              continue
                        data = np.asarray(roi_entry[hemi])
                        mask = data > 0
                        if hemi not in hemisphere_values:
                              hemisphere_values[hemi] = np.zeros(data.shape[0], dtype=np.int32)
                        hemisphere_values[hemi][mask] = code

            self.add_additional_ROI_value_map(
                  key=merged_key,
                  hemisphere_values=hemisphere_values,
                  value_map=value_map,
                  color_map=color_map,
                  overwrite=overwrite
            )
            return merged_key

      def mergeROIs(
            self,
            subroi_keys: list[Hashable],
            merged_key: Hashable,
            overwrite: bool=False
      ) -> Hashable:
            """Merge multiple ROI keys into one binary ROI."""
            if len(subroi_keys) == 0:
                  raise ValueError("subroi_keys must contain at least one ROI key.")

            hemisphere_values: dict[str, np.ndarray] = {}
            value_map: dict[int, str] = {0: 'non-indexed', 1: str(merged_key)}
            color_map: dict[str, tuple[float, float, float, float]] = {
                  'non-indexed': (0.1, 0.1, 0.1, 0.0),
                  str(merged_key): (1.0, 0.0, 0.0, 1.0),
            }

            for subkey in subroi_keys:
                  if subkey not in self.additional_ROIs:
                        raise KeyError(f"SubROI '{subkey}' not found in additional_ROIs.")
                  roi_entry = self.additional_ROIs[subkey]
                  if not isinstance(roi_entry, dict):
                        raise ValueError(f"SubROI '{subkey}' is not stored as a hemisphere-valued dictionary.")

                  roi_vmap = roi_entry.get('vmap', {})
                  roi_cmap = roi_entry.get('cmap', {})
                  positive_labels = [roi_vmap[value] for value in roi_vmap if value != 0]
                  if len(positive_labels) > 0 and positive_labels[0] in roi_cmap:
                        color_map[str(merged_key)] = roi_cmap[positive_labels[0]]

                  for hemi in ['lh', 'rh']:
                        if hemi not in roi_entry:
                              continue
                        data = np.asarray(roi_entry[hemi])
                        mask = data > 0
                        if hemi not in hemisphere_values:
                              hemisphere_values[hemi] = np.zeros(data.shape[0], dtype=np.int32)
                        hemisphere_values[hemi][mask] = 1

            self.add_additional_ROI_value_map(
                  key=merged_key,
                  hemisphere_values=hemisphere_values,
                  value_map=value_map,
                  color_map=color_map,
                  overwrite=overwrite
            )
            return merged_key
            
      def flatplot_additional_ROI(self,key: Hashable, side: str, ax:Optional[Axes]=None,showLegend:bool=False,outline:bool=False,opacity:float=1.0,color_override: tuple=None)->Axes:
            """Overlay ROI values on a flatmap axis."""
            hemi = self._side2hemi(side)
            if not hemi in self.additional_ROIs[key]:
                  return None
            import matplotlib.patches as mpatches
            if ax == None:
                  ax = self.flatmap_plot(side=side,annot=False,outline=False)
            data:np.ndarray = self.additional_ROIs[key][hemi]
            vmap =  self.additional_ROIs[key]['vmap']
            cmap_dict =  self.additional_ROIs[key]['cmap'].copy()
            if color_override is not None:
                  for i in vmap.keys():
                        if i > 0:
                              cmap_dict[vmap[i]] = color_override

            flat_geom, _, _ =  self._load_surface_maps(side)
            
            
            verts_xy = flat_geom.agg_data('pointset')[:,0:2]
            faces: np.ndarray = flat_geom.agg_data('triangle')
            if outline:
                  """label_patches handles all the legend and formatting, so just this call is sufficient"""
                  id_cmap = {v: cmap_dict[vmap[v]] for v in np.unique(data) if v in vmap}
                  legend = {v: vmap[v] for v in id_cmap} if showLegend else None
                  label_patches(
                        ax=ax,
                        verts_xy=verts_xy,
                        faces=faces,
                        labels=data,
                        cmap=id_cmap,
                        outline=True,
                        opacity=opacity,
                        legend=legend
                  )
                  return ax

            tri = mtri.Triangulation(verts_xy[:,0], verts_xy[:,1], faces)
            unique = np.unique(data)
            unique = unique[unique != 0]
            proxy_patch = []
            for v in unique:
                  mask = (data == v)
                  tri_mask = np.all(mask[faces], axis=1)
                  tri_local = mtri.Triangulation(tri.x, tri.y, tri.triangles.copy())
                  tri_local.set_mask(~tri_mask)
                  if not np.any(mask):
                        continue
                  z = np.zeros_like(data, dtype=float)
                  z[mask] = 1.0
                  color = cmap_dict[vmap[v]]
                  lab = vmap[v]
                  rgba = (color[0], color[1], color[2], (color[3] if len(color) > 3 else 1.0) * opacity)
                  cmap = ListedColormap([
                        (0,0,0,0),
                        rgba
                  ])
                  tricf = ax.tricontourf(
                        tri_local,
                        z,
                        levels=[0.5, 1.5],
                        cmap=cmap,
                        antialiased=False
                  )
                  tricf.set_label(lab)
                  proxy_patch.append(mpatches.Patch(color=rgba, label=lab))
            if showLegend:
                  ax.legend(handles=proxy_patch)
            return ax


      def load_electrodes_from_dat(self,electrodeDir:Path)-> dict:
            """Load electrodes from dat."""
            if not os.path.exists(electrodeDir):
                  raise FileNotFoundError
            if not isinstance(electrodeDir,Path): electrodeDir=Path(electrodeDir)
            electrodes = {}
            for f in electrodeDir.glob("*.dat"):
                  with open(f,'r') as fp:
                        name = f.name.replace('.dat','')
                        lines = fp.readlines()
                        lines = [i.strip() for i in lines]
                        n_points = int(lines[lines.index('info')+1].split(' ')[-1])
                        try:  electrodes.update({f'{name}_{idx+1}':np.array(i.split(),dtype=float) for idx,i in enumerate(lines[1:n_points+1])}) 
                        except ValueError: electrodes.update({f'{name}_{idx+1}':np.array(i.split(),dtype=float) for idx,i in enumerate(lines[0:n_points])}) 
            return electrodes
      def surface_resample(self,fileList: List[Path], process: bool)->Tuple[Path,bool]:
            """Surface resample."""
            for i in fileList:
                  name = i.name
                  save_dir = i.parent.parent / f'fs_LR.{self.atlas.name}/surf'
                  if not os.path.exists(save_dir): process=True
                  os.makedirs(save_dir,exist_ok=True)
                  fs_sphere, hemi, tag = self._checkHemi(name)
                  pt_sphere = i.parent / f'{hemi}.{self.atlas.pt_sphere_name}'
                  outfile = save_dir / f'{self.atlas.name}.{name}'
                  cmd = f"wb_command -surface-resample {i} {pt_sphere} {fs_sphere} BARYCENTRIC {outfile}".split()
                  if process: run_process((cmd))
                  cmd = f"wb_command -set-structure {outfile} {tag}".split()
                  if process: run_process(cmd)
            return save_dir.parent, process
      
      def surface_map_electrodes(self,surfaceDir:Path,electrodes: dict, process:bool, thresh:float=10,bipolar: bool=False)->None:
            """Surface map electrodes."""
            if process: 
                  surf_fps = surfaceDir.glob('surf/*pial.surf.gii')
                  if bipolar:outdir = surfaceDir/'electrodes_bipolar'
                  else:outdir = surfaceDir/'electrodes'
                  os.makedirs(outdir,exist_ok=True)
                  elec_locs = np.asarray(list(electrodes.values()))
                  for i in surf_fps:
                        surface: nibabel.gifti.gifti.GiftiImage = load_nii_file(i)
                        verts = surface.agg_data('pointset')
                        # tris = surface.agg_data('triangle')
                        _,hemi,_ = self._checkHemi(i.name)
                        side = self._hemi2side(hemi)
                        annot_file=load_nii_file(self.annot_file_template_path_fsLR.parent / self.annot_file_template_path_fsLR.name.replace('[hemi]',hemi))
                        flatmap=load_nii_file(self.flatmap_template_path_fsLR.parent / self.flatmap_template_path_fsLR.name.replace('[side]',side)).agg_data()[0]
                        annot_data = annot_file.agg_data()
                        annot_labels = {i.key:i.label for i in annot_file.labeltable.labels}
                        if len(elec_locs > 0):
                              distances,vert_idx = pdist2(elec_locs,verts,num_mins=1)
                              vert_mapping = {a:dict(vert=int(c),dist=float(b[0]),label=int(annot_data[c]),region=annot_labels[annot_data[c]],flatcoords=[float(i) for i in flatmap[c]]) for a,b,c in zip(electrodes,distances,vert_idx) if b < thresh}
                              outfile = outdir / f'{hemi}.{self.atlas.name}.electrodes.json'
                              with open(outfile, 'w') as fp:
                                    json.dump(vert_mapping,fp)
                        else: print(f'{hemi}, {i.name} has no electrodes')
            else: pass
      
      def _side2hemi(self,side:str)->str:
            """ side2hemi."""
            if side.lower() == 'l': return 'lh'
            elif side.lower() == 'r': return 'rh'
            else: 
                  raise KeyError(f'key {side} is not of acceptable format of L/R for side key')
      def _hemi2side(self,side:str)->str:
            """ hemi2side."""
            return side[0].upper()
      
      def metric_resample(self,fileList: List[Path], process: bool)->None:
            """Metric resample."""
            for i in fileList:
                  name = i.name
                  save_dir = i.parent.parent / f'fs_LR.{self.atlas.name}/metric'
                  os.makedirs(save_dir,exist_ok=True)
                  fs_sphere, hemi, tag = self._checkHemi(name)
                  fs_lr_hemi = hemi[0].upper()
                  pt_sphere = i.parent.parent / f'surf/{hemi}.{self.atlas.pt_sphere_name}'
                  fs_midthickness = self.atlas.source / f'resample_fsaverage/fs_LR.{fs_lr_hemi}.midthickness_va_avg.{self.atlas.name}_fs_LR.shape.gii'
                  pt_midthickness = i.parent.parent / f'surf/{hemi}.midthickness.surf.gii'
                  outfile = save_dir / f'{self.atlas.name}.{name}'
                  # cmd = f"wb_command -metric-resample {i} {pt_sphere} {fs_sphere} ADAP_BARY_AREA {outfile} -area-metrics {pt_midthickness} {fs_midthickness}".split()
                  cmd = f"wb_command -metric-resample {i} {pt_sphere} {fs_sphere} BARYCENTRIC {outfile}".split()
                  if process: run_process((cmd))
                  cmd = f"wb_command -set-structure {outfile} {tag}".split()
                  if process: run_process(cmd)
      def label_resample(self,fileList: List[Path],process:bool)->None:
            """Label resample."""
            for i in fileList:
                  name = i.name
                  save_dir = i.parent.parent / f'fs_LR.{self.atlas.name}/label'
                  os.makedirs(save_dir,exist_ok=True)
                  fs_sphere, hemi, tag = self._checkHemi(name)
                  fs_lr_hemi = hemi[0].upper()
                  pt_sphere = i.parent.parent / f'surf/{hemi}.{self.atlas.pt_sphere_name}'
                  fs_midthickness = self.atlas.source / f'resample_fsaverage/fs_LR.{fs_lr_hemi}.midthickness_va_avg.{self.atlas.name}_fs_LR.shape.gii'
                  pt_midthickness = i.parent.parent / f'surf/{hemi}.midthickness.surf.gii'
                  outfile = save_dir / f'{self.atlas.name}.{name}'
                  # cmd = f"wb_command -metric-resample {i} {pt_sphere} {fs_sphere} ADAP_BARY_AREA {outfile} -area-metrics {pt_midthickness} {fs_midthickness}".split()
                  cmd = f"wb_command -label-resample {i} {pt_sphere} {fs_sphere} BARYCENTRIC {outfile}".split()
                  if process: run_process(cmd)
                  cmd = f"wb_command -set-structure {outfile} {tag}".split()
                  if process: run_process(cmd)
                  annot = load_nii_file(outfile)
                  label_dict = {int(i.key):str(i.label) for i in annot.labeltable.labels}
                  dict_fp = save_dir / f'{self.atlas.name}.{name.replace('label.gii','json')}'
                  with open(dict_fp,'w') as fp:
                        json.dump(label_dict,fp)
      
      def _checkHemi(self,fname)-> Tuple[Path,str,str]:      
            """  checkhemi."""
            side = fname.find('rh.') > -1
            if side:
                  sphere = self.atlas.right_sphere
                  hemi = 'rh'
                  tag = 'CORTEX_RIGHT'
            else:
                  sphere = self.atlas.left_sphere
                  tag = 'CORTEX_LEFT'
                  hemi = 'lh'
            return sphere,hemi,tag
      
      def fs_seg_to_gifti(self,seg_path,out_path,process: bool, correct_affine:bool)->Tuple[List[Path],List[Path],List[Path]]:
            """Fs seg to gifti."""
            from nibabel import affines as aff
            if not os.path.exists(out_path/'surf'):
                  process = True
            os.makedirs(out_path/'surf',exist_ok=True)
            os.makedirs(out_path/'label',exist_ok=True)
            os.makedirs(out_path/'func',exist_ok=True)
            sides = ['l','r']
            # surf_targets = ['pial','white','inflated','sphere.reg']
            surf_targets = ['pial','white','sphere.reg',"sphere"]
            metric_targets = ['sulc','curv']
            label_targets = ['aparc.a2009s.annot']
            out_surfs = []
            out_metrics = []
            out_labels = []
            for side in sides:
                  for surf in surf_targets:
                        if not os.path.exists(seg_path/'surf'/f'{side}h.{surf}'):
                              cmd = ["mris_convert",seg_path/'surf'/f'{side}h.{surf}.T1',out_path/'surf'/f'{side}h.{surf}.surf.gii']
                        else:
                              cmd = ["mris_convert",seg_path/'surf'/f'{side}h.{surf}',out_path/'surf'/f'{side}h.{surf}.surf.gii']
                              cmd1 = ["mris_convert",seg_path/'surf'/f'{side}h.{surf}.T1',out_path/'surf'/f'{side}h.{surf}.surf.gii']
                        if process:
                              try:
                                    run_process([str(i) for i in cmd])
                              except subprocess.CalledProcessError as e:
                                    print(e)
                                    print('retrying with pial.T1')
                                    run_process([str(i) for i in cmd1])
                                    
                        out_surfs.append(out_path/'surf'/f'{side}h.{surf}.surf.gii')
                        if not 'sphere' in surf and correct_affine and process:
                              surface: GiftiImage = nibabel.load(out_path/'surf'/f'{side}h.{surf}.surf.gii') # type: ignore
                              affine = surface.darrays[0].coordsys.xform
                              newVerts = aff.apply_affine(affine,surface.agg_data('pointset'))
                              surface.darrays[0].data = newVerts
                              surface.darrays[0].coordsys.xform = np.eye(4)
                              nibabel.loadsave.save(surface,out_path/'surf'/f'{side}h.{surf}.surf.gii')
                  cmd = f"wb_command -surface-average {out_path}/surf/{side}h.midthickness.surf.gii -surf {out_path}/surf/{side}h.white.surf.gii -surf {out_path}/surf/{side}h.pial.surf.gii".split(' ')
                  _,_,tag = self._checkHemi(f'{side}h.midthickness.surf.gii')
                  cmd1 = f"wb_command -set-structure {out_path}/surf/{side}h.midthickness.surf.gii {tag}".split()
                  if process: run_process(cmd); run_process(cmd1) 
                  out_surfs.append(out_path/f"surf/{side}h.midthickness.surf.gii")
                  
                  for metric in metric_targets:
                        cmd = ["mris_convert", "-c",seg_path/'surf'/f'{side}h.{metric}',seg_path/'surf'/f'{side}h.white',out_path/'func'/f'{side}h.{metric}.func.gii']
                        if process:
                              run_process([str(i) for i in cmd])
                        out_metrics.append(out_path/'func'/f'{side}h.{metric}.func.gii')
                  for label in label_targets:
                        cmd = ["mris_convert", "--annot",seg_path/'label'/f'{side}h.{label}',seg_path/'surf'/f'{side}h.white',out_path/'label'/f'{side}h.{label}.label.gii']
                        if process:
                              run_process([str(i) for i in cmd])
                        out_labels.append(out_path/'label'/f'{side}h.{label}.label.gii')
            return out_surfs, out_metrics, out_labels

      def flatmap_export(self,outpath)->None:
            """Flatmap export."""
            os.makedirs(outpath,exist_ok=True)
            # LUT = load_fs_LUT()
            # cmap = {i:(j[2:5]) for i,j in LUT.items()}
            import matplotlib.tri as mtri
            for hemi in ['L','R']:
                  if hemi == 'L':
                        side = 'lh'
                  else:
                        side = 'rh'
                  flat_geom =load_nii_file(self.atlas.source / f'colin.cerebral.{hemi}.flat.{self.atlas.name}_fs_LR.surf.gii')
                  sulcus_map=load_nii_file(self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}' / 'metric' / f'{self.atlas.name}.{side}.sulc.func.gii').agg_data()
                  annot_file =load_nii_file(self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}' / 'label' / f'{self.atlas.name}.{side}.aparc.a2009s.annot.label.gii')
                  annot_data = annot_file.agg_data()
                  color_labels = annot_file.labeltable
                  cmap= {}
                  for label in color_labels.labels:
                        label_name = label.key
                        # Color values are 0-255, stored as float in nibabel, so convert to int
                        green = int(label.green * 255)
                        red = int(label.red * 255)
                        blue = int(label.blue * 255)
                        alpha = int(label.alpha * 255)
                        cmap[label_name] = (red, green, blue, alpha)
                  with open(self.data_root /'pt-space'/ f'fs_LR.{self.atlas.name}' / 'electrodes' / f'{side}.{self.atlas.name}.electrodes.json','r') as fp:
                        electrodes = json.load(fp)
                  verts_xy = flat_geom.agg_data('pointset')[:,0:2]
                  faces: np.ndarray = flat_geom.agg_data('triangle')
                  vals = sulcus_map
                  
                        # import pyvista as pv
                        # mesh = pv.PolyData(flat_geom.agg_data('pointset'),faces_flat)
                        # ax = pv.Plotter()
                        # ax.add_mesh(mesh)
                        # ax.show_axes()
                        # ax.view_xy()
                        # ax.show()
            plt.show()

      
      def parse_electrodes_from_names(self,electrodes: dict|pd.DataFrame,subsetNames:List[str]|Path|pd.Index)-> Tuple[dict,list]:
            """
            parse_electrodes_from_names: 
            slices electrode dictionary from a list of channel names or a csv file with at least one column labeled 'electrodes'

            Args:
                  electrodes (dict | DataFrame): dictionary of electrodes generated by load_flat_electrodes or a df from groupAtlas
                  subsetNames (List[str] | Path): List of hashable keys to segment electrodes by, or a filepath to a csv containing hashable keys in a single column titled 'electrodes'

            Returns:
                  dict: subset of dictionary of electrodes generated by load_flat_electrodes based on subsetNames
            """
            
            if isinstance(subsetNames,str|Path):
                  f = subsetNames
                  with open(f,'r') as fp:
                        df = pd.read_csv(fp) 
                  subsetNames = df['electrodes'].to_list()
            if isinstance(electrodes,dict):
                  output = {i:electrodes[i] for i in subsetNames if i in electrodes}
            elif isinstance(electrodes,pd.DataFrame) and isinstance(subsetNames,list):
                  idxs = electrodes.index[electrodes['channel'].isin(subsetNames)]
                  output = electrodes.loc[idxs]
            elif isinstance(electrodes,pd.DataFrame) and isinstance(subsetNames,pd.Index):
                  output = electrodes.loc[subsetNames]
                  subsetNames = subsetNames.to_list()
            else:
                  print('incorrect electrode type, must be dict|DataFrame')
                  return electrodes
            return output, subsetNames
      
      def parse_electrodes_from_regions(self,electrodes:dict|pd.DataFrame, regions:List[str])-> dict:
            """
            parse_electrodes_from_regions
            slices electrode dictionary from a list of region names

            Args:
                  electrodes (dict): dictionary of electrodes generated by load_flat_electrodes
                  subsetNames (List[str]): List of hashable keys to segment electrode regions by, must be taken from the annotation tabel associated with this data (typically the aparc.2009 atlas)

            Returns:
                  dict: subset of dictionary of electrodes generated by load_flat_electrodes based on subsetRegions
            """
            if isinstance(electrodes,dict):
                  output = {i:j for i,j in electrodes.items() if j['region'] in regions}
            else: 
                  idxs = electrodes.index[electrodes['region'].isin(regions)]
                  output = electrodes.loc[idxs]
            return output
      
      
      def _load_surface_maps(self,side)->Tuple[GiftiImage,GiftiImage,GiftiImage]:
            """
            _load_surface_maps _summary_

            Args:
                  hemi (_type_): _description_

            Returns:
                  tuple: returns flat geometry (points and verts),sulcal depth map, and annotation labels
            """
            hemi = self._side2hemi(side)
            
            fp = self.flatmap_template_path_fsLR.parent / self.flatmap_template_path_fsLR.name.replace('[side]',side)
            flat_geom =load_nii_file(fp)
            fp = self.sulcal_depth_map_template_path_fsLR.parent / self.sulcal_depth_map_template_path_fsLR.name.replace('[hemi]',hemi)
            sulcus_map =load_nii_file(fp).agg_data()
            fp = self.annot_file_template_path_fsLR.parent / self.annot_file_template_path_fsLR.name.replace('[hemi]',hemi)
            annot_file=load_nii_file(fp)
            return flat_geom,sulcus_map,annot_file
      def flatmap_plot(self, side: str='L', targetRegions: Optional[list]=None, annot: bool=True, outline: bool=True, segOpacity=0.2, legend: bool=False, ax: Optional[Axes]=None) -> Axes:
            """
            flatmap_plot plot a flatmap in 2D with the associated kwargs. 
            Uses the default freesurfer color LUT.

            Args:
                  hemi (str, optional): 'L' or 'R' for hemisphere of interest. Defaults to 'L'.
                  targetRegions (Optional[list], optional): list of parcels to draw outlines/color in. Defaults to None.
                  annot (bool, optional): flag to draw parcel labels. Defaults to True.
                  outline (bool, optional): flag to draw parcel labels as outlines. Setting this to false and annot to true will yield fully shaded parcels. Defaults to True.
                  segOpacity (float, optional): opacity of parcels. Defaults to 0.2.
                  legend (bool, optional): plot parcel legend to indicate color maps. Defaults to False.
                  ax (Optional[Axes], optional): axis to draw onto. Creates a new axis when None.

            Returns:
                  Axes: matplotlib Axes object with flatmap on it
            """
            hemi = self._side2hemi(side) 
            # flat_geom =load_gifti(self.map_template_path.parent / self.map_template_path.name.replace('[hemi]',hemi))
            # sulcus_map =load_gifti(self.sulcus_map_template_path.parent / self.sulcus_map_template_path.name.replace('[side]',side)).agg_data()
            # annot_file=load_gifti(self.annot_file_template_path.parent / self.annot_file_template_path.name.replace('[side]',side))
            flat_geom,sulcus_map,annot_file = self._load_surface_maps(side=side)
            annot_data = np.asarray(annot_file.agg_data()).copy()
            gifti_labels = annot_file.labeltable
            # cmap= {}
            # region_map={}
            # for label in gifti_labels.labels:
            #       label_name = label.key
            #       # Color values are 0-255, stored as float in nibabel, so convert to int
            #       green = int(label.green * 255)
            #       red = int(label.red * 255)
            #       blue = int(label.blue * 255)
            #       alpha = int(label.alpha * 255)
            #       cmap[label_name] = (red, green, blue, alpha)
            #       region_map[label_name] = label.label
            cmap,region_map = annotations_from_gifti_labels(gifti_labels)

            if targetRegions is not None:
                  if isinstance(targetRegions,str): targetRegions=[targetRegions]
                  target_ids: set[int] = set()
                  target_names = {str(i) for i in targetRegions}
                  for label_id, region_name in region_map.items():
                        if region_name in target_names or str(label_id) in target_names:
                              target_ids.add(int(label_id))
                  annot_data[~np.isin(annot_data, list(target_ids))] = -1
            
            verts_xy = flat_geom.agg_data('pointset')[:,0:2]
            faces: np.ndarray = flat_geom.agg_data('triangle')
            vals = sulcus_map*-1
            n_vert = verts_xy.shape[0]
            neighbors = [[] for _ in range(n_vert)]
            
            tri = mtri.Triangulation(verts_xy[:,0], verts_xy[:,1], faces)
            pad = 2.0  # mm border
            xmin, xmax = verts_xy[:,0].min()-pad, verts_xy[:,0].max()+pad
            ymin, ymax = verts_xy[:,1].min()-pad, verts_xy[:,1].max()+pad
            W = 2048
            H = int((ymax-ymin)/(xmax-xmin) * W)
            if ax is None:
                  _, ax = plt.subplots(figsize=(8, 8 * (H / W)))
            ax.tricontourf(tri, vals, levels=201, cmap='Greys_r', zorder=0)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.figure.tight_layout()
            if annot:
                  edges = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])
                  edges.sort(axis=1)
                  edges = np.unique(edges, axis=0)
                  neighbors = [[] for _ in range(len(verts_xy))]
                  for a,b in edges:
                        neighbors[a].append(b)
                        neighbors[b].append(a)

                  labels=annot_data
                  if legend:
                        legend_key = {label_id: label_name for label_id, label_name in region_map.items() if label_id in np.unique(labels)}
                  else: legend_key = None
                  label_patches(ax,verts_xy,faces,labels,cmap,outline=outline,opacity=segOpacity,legend=legend_key)
            return ax
      
      def flatmap_effect_plot(self,hemi: str='L',targetRegions: Optional[list]=None,annot:bool=True,outline: bool=True, segOpacity=0.2,legend:bool=False)->Axes:
            
            
            
            """Flatmap effect plot."""
            pass
      
      def effect_class_surface_heat_diffusions(
            self,
            electrodes: pd.DataFrame,
            diffusion: float = 0.01,
            effect: Optional[Hashable] = None
            ) -> dict:

            """Effect class surface heat diffusions."""
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            classes = list(np.unique(electrodes['class'].to_list()))
            class_labels = {classes[i]: i for i in range(len(classes))}
            electrodes['class_label'] = electrodes['class'].map(class_labels)

            if 'na' in classes:
                  classes.pop(classes.index('na'))

            def mass_matrix(V, F)->np.matrix:
                  """Mass matrix."""
                  N = len(V)
                  area = np.zeros(N)

                  for tri in F:
                        i, j, k = tri
                        a = V[j] - V[i]
                        b = V[k] - V[i]
                        tri_area = 0.5 * np.linalg.norm(np.cross(a, b))

                        area[i] += tri_area / 3.0
                        area[j] += tri_area / 3.0
                        area[k] += tri_area / 3.0
                  return sp.diags(area)

            out = {}
            for side in ['lh', 'rh']:
                  surface = load_nii_file(
                        self.midthickness_template_path_fsLR.parent /
                        self.midthickness_template_path_fsLR.name.replace('[side]', side))
                  V, F = surface.agg_data()
                  L = cotangent_laplacian(V, F)
                  M = mass_matrix(V, F)
                  A = (M - diffusion * L).tocsr()
                  solver = spla.factorized(A)
                  df = electrodes.query("hemi==@side")
                  class_dict = {}
                  for c in classes:
                        delta = np.zeros(len(V))
                        ID = class_labels[c]
                        temp = df.query("class_label==@ID")
                        if len(temp) == 0:
                              class_dict[c] = np.zeros(len(V))
                              continue
                        idxs = temp['vert'].to_numpy(dtype=int)
                        if effect is not None:
                              vals = temp[effect].to_numpy()
                              # robust normalization (0 → 1)
                              vals = vals - vals.min()
                              denom = vals.max() + 1e-8
                              vals = vals / denom
                              np.add.at(delta, idxs, vals)
                        else:
                              np.add.at(delta, idxs, 1.0)
                        rhs = M @ delta
                        F_c = solver(rhs)
                        F_c[F_c < 0] = 0

                        if F_c.max() > 0:
                              F_c /= F_c.max()
                        class_dict[c] = F_c
                  out[side] = class_dict

            return out
                  
      
      def effect_class_surface_geodesic_gaussian(
            self,
            electrodes: pd.DataFrame,
            sigma: float = 10.0,
            effect: Optional[Hashable] = None
            ) -> dict:

            """Effect class surface geodesic gaussian."""
            import numpy as np
            import gdist  # pip install gdist

            classes = list(np.unique(electrodes['class'].to_list()))
            class_labels = {classes[i]: i for i in range(len(classes))}
            electrodes['class_label'] = electrodes['class'].map(class_labels)

            if 'na' in classes:
                  classes.pop(classes.index('na'))
            out = {}
            for hemi in ['lh', 'rh']:
                  side = self._hemi2side(hemi)
                  surface = load_nii_file(
                        self.midthickness_template_path_fsLR.parent /
                        self.midthickness_template_path_fsLR.name.replace('[hemi]', hemi)
                  )

                  V, F = surface.agg_data()
                  df = electrodes.query("hemi==@hemi")
                  class_dict = {}
                  for c in classes:
                        ID = class_labels[c]
                        temp = df.query("class_label==@ID")
                        if len(temp) == 0:
                              class_dict[c] = np.zeros(len(V))
                              continue
                        idxs = temp['vert'].to_numpy(dtype=int)
                        if effect is not None:
                              vals = temp[effect].to_numpy()
                              vals = vals - vals.min()
                              vals /= (vals.max() + 1e-8)
                        else:
                              vals = np.ones(len(idxs))

                        F_c = np.zeros(len(V))
                        dist = gdist.compute_gdist(
                              V.astype(np.float64),
                              F.astype(np.int32),
                              source_indices=idxs.astype(np.int32))
                        F_c = np.exp(-(dist**2) / (2 * sigma**2))
                        class_dict[c] = F_c
                  out[hemi] = class_dict
            return out
      
      def plot_single_effect_class_diffusion(self,class_dict:dict, class_name:Hashable, color: tuple=(1,0,0),normalize:bool=True, scale:str='linear',side: str='L',targetRegions: Optional[list]=None,annot:bool=True,outline: bool=True, segOpacity=0.2,legend:bool=False,ax: Axes=None, render='tri',image_res=1000)->Axes:
            """Plot single effect class diffusion."""
            from scipy.ndimage import gaussian_filter
            hemi = self._side2hemi(side)
            if not  class_name in class_dict[hemi]:
                  return ax
            effect = class_dict[hemi][class_name]
            if ax is None:
                  ax = self.flatmap_plot(side,targetRegions,annot,outline,segOpacity,legend)
            if normalize:
                  effect /= effect.max() + 1e-8
            
            fp=self.flatmap_template_path_fsLR.parent / self.flatmap_template_path_fsLR.name.replace('[side]',side)
            surface = load_nii_file(fp)
            verts,tris = surface.agg_data()
            verts = verts[:,0:-1]
            effect_smoothed = gaussian_filter(effect,0.1)
            rgba_v = np.zeros([len(effect),4])
            rgba_v[:,:3] = color
            rgba_v[:,-1] = effect
            rgba_f = rgba_v[tris].mean(axis=1)
            
            tri = mtri.Triangulation(verts[:,0],verts[:,1],tris)
            # cmap = alpha_colormap(color)
            cmap = alpha_colormap(color)
            # ax.tripcolor(tri,effect,shading='gouraud',cmap='hot',edgecolors='none',alpha=0.5)
            # pc = ax.tripcolor(tri,effect,cmap=cmap,edgecolors='none',shading='flat',alpha=0.99)
            # pc.set_facecolor(rgba_f)
            # pc.set_alpha(None)
            # pc.set_linewidth(0)
            # pc.set_edgecolor('none')
            if render == 'tri':
                  from matplotlib.collections import PolyCollection
                  polys = verts[tris]  # (M,3,2)
                  pc = PolyCollection(polys, facecolors=rgba_f, edgecolors='none')
                  pc.set_alpha(None)
                  pc.set_linewidth(0)
                  pc.set_edgecolor('none')
                  ax.add_collection(pc)
            else:
                  from scipy.interpolate import griddata
                  xmin, xmax = verts[:,0].min(), verts[:,0].max()
                  ymin, ymax = verts[:,1].min(), verts[:,1].max()

                  xi = np.linspace(xmin, xmax, image_res)
                  yi = np.linspace(ymin, ymax, image_res)
                  XI, YI = np.meshgrid(xi, yi)
                  ZI = griddata(verts, effect, (XI, YI), method='nearest')
                  ZI = np.nan_to_num(ZI)
                  # pmax = np.percentile(ZI, 99)
                  # ZI = np.clip(ZI / (pmax + 1e-8), 0, 1)
                  img = np.zeros((image_res, image_res, 4))
                  img[..., :3] = color
                  img[..., 3] = ZI
                  ax.imshow(
                        img,
                        origin='lower',
                        extent=[xmin, xmax, ymin, ymax],
                        interpolation='bilinear'
                  )

                  ax.set_aspect('equal')
                  ax.axis('off')
            return ax
            
      
                  
            
            
            
      def plot_both_maps(self,targetRegions: Optional[list]=None,outline: bool=True, segOpacity=0.2)-> list:
            """Plot both maps."""
            axs=[]
            for hemi in ['L','R']:
                  axs.append(self.flatmap_plot(side=hemi,targetRegions=targetRegions,outline=outline,segOpacity=segOpacity))
            return axs
      
      def plot_electrodes(self,ax: Axes,hemi:str,bipolar:bool,subset:Optional[list]=None,color:dict|tuple=(0,0,0,1),size:dict|float=5,name_mapping:Optional[dict]=None)->Axes:
            """Plot electrodes."""
            electrodes = self.load_flat_electrodes(hemi,bipolar)
            if name_mapping is not None:
                  electrodes = {name_mapping[i]:j for i,j in electrodes.items()}
            else:
                  electrodes,_ = electrode_name_match(electrodes)
            if subset is not None: 
                  if subset[0]=='regions':
                        electrodes,_ = self.parse_electrodes_from_regions(electrodes,subset[1])
                  if subset[0]=='names':
                        electrodes,_ = self.parse_electrodes_from_names(electrodes,subset[1])
            if not isinstance(size,dict): 
                  size = {i:size for i in electrodes}
            if not isinstance(color, dict): 
                  color = {i:color for i in electrodes}
            for k,v in electrodes.items():
                  coords = v['flatcoords']
                  ax.scatter(coords[0],coords[1],color=color[k],s=size[k],zorder=10)
            
            return ax
      
      
      def plot_network(self,AX: Axes, hemi: str, network:networkGraph, regionSubset: List[str]=[],channelSubset:List[str]=[])->Axes:
            """Plot network."""
            flat_electrodes = self.load_flat_electrodes(hemi,True)
            flat_electrodes, name_mapping = electrode_name_match(flat_electrodes,list(network._adj.keys()))
            if bool(channelSubset):
                  flat_electrodes,keys = self.parse_electrodes_from_names(flat_electrodes,channelSubset)
            elif bool(regionSubset):
                  flat_electrodes,keys = self.parse_electrodes_from_regions(flat_electrodes,regionSubset)
            else:
                  pass
            subset = [i for i in network._adj if bool(network._adj[i]) and i in flat_electrodes.keys()]
            segments:np.ndarray = self.__get_segments(flat_electrodes,network,subset)
            # segments = self.__cable_routing(segments,grid_res=2,padding=5)
            collec = LineCollection(segments,zorder=3) # type: ignore
            self.plot_electrodes(AX,hemi,subset=['names',keys],bipolar=True,name_mapping=name_mapping,size=20)
            AX.add_collection(collec)
            self.fit_image_to_electrode(AX,flat_electrodes)
            print('cheese')
            # plt.show()
            return AX
      
      def fit_image_to_electrode(self,ax: Axes,electrodes: dict,padding:float = 25)-> Axes:
            """Fit image to electrode."""
            xmin,xmax,ymin,ymax = self.get_electrode_bbox(electrodes)
            xmin-=padding; xmax+=padding; ymin-=padding; ymax+=padding
            ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax))
            return ax
      
      def get_electrode_bbox(self,electrodes)-> tuple:
            """Get electrode bbox."""
            coords = np.array([[i['flatcoords'][0],i['flatcoords'][1]] for i in electrodes.values()])
            xmin,ymin = np.min(coords,axis=0)      
            xmax,ymax = np.max(coords,axis=0)      
            return xmin,xmax,ymin,ymax
      
      def __get_spline_waypoints(self,electrodes,n_waypoints)->None:
            #TODO: implement. I knew I had shifted to spline based cable routing at some point but evidently never finished it. feature coming hehe.
            """  get spline waypoints."""
            pass



      def __cable_routing(self,segments:np.ndarray,grid_res: float,padding:int):
            """  cable routing."""
            all_pts = np.array([p for seg in segments for p in seg])
            diffs = segments[:,1,:] - segments[:,0,:]
            distances = np.linalg.norm(diffs,axis=1)
            distance_sort = np.argsort(distances)
            segs_sorted = segments[distance_sort[::-1]]
            # all_pts
            xmin,ymin = all_pts.min(axis=0)
            xmax,ymax = all_pts.max(axis=0)
            pad = grid_res*padding
            xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
            W,H = int((xmax-xmin)/grid_res), int((ymax-ymin)/grid_res)
            dx = (xmax - xmin) / (W - 1)
            dy = (ymax - ymin) / (H - 1)
            # pairs_grid = self.prepare_grid(segs_sorted,H,W,xmin,ymin,dx,dy)
            # routed, occupancy = self.traverse_grid(pairs_grid,H,W)
            
            def to_data(gp: Tuple[float,float]):
                  """To data."""
                  gx, gy = gp
                  x = xmin + gx * dx
                  y = ymin + gy * dy
                  return (x, y)

            
            return segs_sorted
      def traverse_grid(self,pairs_grid,H,W)->Tuple[list,np.ndarray]:
            #TODO: figure out what the fuck I was trying to do here. I know this was for cable routing to show connections but I think doing a weight spline makes much more sense. 
            """Traverse grid."""
            w_L   = 1.0     # base step cost
            w_S   = 25.0    # soft obstacle (prior routes)
            w_E   = 8.0     # endpoint repulsion (other pins)
            sigma_path = 3  # blur radius (cells) for routed paths
            sigma_pin  = 2  # blur radius (cells) for endpoint repel
            occ = np.zeros((H,W),float)
            pins_mask = np.zeros((H,W),float) 
            for s,g in pairs_grid:
                  pins_mask[s[1],s[0]]=1.0
                  pins_mask[g[1],g[0]]=1.0
                  
            moves = [ (1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
                  (1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(-1,-1,np.sqrt(2)) ]
            routed_paths = []  # list of grid polylines
            for idx, (s, g) in enumerate(pairs_grid):
                  # Soft obstacle from already routed paths
                  C_soft = gaussian_filter(occ, sigma=sigma_path)
                  # Endpoint repel: exclude current pair’s pins
                  pins = pins_mask.copy()
                  pins[s[1], s[0]] = 0.0
                  pins[g[1], g[0]] = 0.0
                  C_end = gaussian_filter(pins, sigma=sigma_pin)
                  # Total per-cell additive cost baseline
                  C = w_L*1.0 + w_S*C_soft + w_E*C_end
                  # A* search
                  def h(p):  # Euclidean heuristic in grid units
                        """H."""
                        return np.hypot(p[0]-g[0], p[1]-g[1])
                  gscore = np.full((H, W), np.inf, float)
                  came   = np.full((H, W, 2), -1, int)
                  openq = []
                  gscore[s[1], s[0]] = 0.0
                  heappush(openq, (h(s), 0.0, s))  # (f, g, (x,y))

                  visited = np.zeros((H, W), bool)
                  while openq:
                        fcur, gcur, u = heappop(openq)
                        if visited[u[1], u[0]]:
                              continue
                        visited[u[1], u[0]] = True
                        if u == g:
                              break
                        for dxm, dym, step_len in moves:
                              vx = u[0] + dxm; vy = u[1] + dym
                              if not (0 <= vx < W and 0 <= vy < H):
                                    continue
                              # step cost = length + destination cell cost
                              step_cost = step_len + C[vy, vx]
                              ng = gcur + step_cost
                              if ng < gscore[vy, vx]:
                                    gscore[vy, vx] = ng
                                    came[vy, vx] = (u[0], u[1])
                                    heappush(openq, (ng + h((vx, vy)), ng, (vx, vy)))
                  # Backtrack path
                  path = [g]
                  u = g
                  while (u[0] != s[0]) or (u[1] != s[1]):
                        pu = came[u[1], u[0]]
                        if pu[0] < 0:
                              # fallback: no path found, use straight line Bresenham
                              x0,y0 = s; x1,y1 = g
                              n = int(max(abs(x1-x0), abs(y1-y0))+1)
                              xs = np.linspace(x0, x1, n).astype(int)
                              ys = np.linspace(y0, y1, n).astype(int)
                              path = list(zip(xs, ys))
                              break
                        u = (pu[0], pu[1])
                        path.append(u)
                  path.reverse()
                  routed_paths.append(np.array(path, int))
                  # Update occupancy with this path (rasterize)
                  for (x,y) in path:
                        occ[y, x] = 1.0
            return routed_paths, occ
      
      @staticmethod
      def prepare_grid(segments:np.ndarray,H:int,W:int,xmin:float,ymin:float,dx:float,dy:float):
            """Prepare grid."""
            def to_grid(p):
                  """To grid."""
                  x, y = p
                  gx = int(round((x - xmin) / dx))
                  gy = int(round((y - ymin) / dy))
                  return (np.clip(gx, 0, W - 1), np.clip(gy, 0, H - 1))
            pairs_grid = [(to_grid(s), to_grid(g)) for (s, g) in segments]
            return pairs_grid
      
      def __get_segments(self,electrodes:dict ,network: networkGraph, channels: list)->np.ndarray:
            """  get segments."""
            starts,ends = [],[]
            for i in channels:
                  if i in electrodes:
                        start = electrodes[i]['flatcoords'][0:-1]
                        entry = network._adj[i]
                        end = np.asarray([electrodes[j]['flatcoords'][0:-1] for j in entry if (j in electrodes and j in channels)])
                        ends.extend(end)
                        starts.extend(np.tile(start,(len(end),1)))
            segs = np.column_stack([starts,ends]).reshape(-1,2,2)
            unique_segs = np.unique(segs,axis=0)
            return unique_segs
      
      def plot_effect_on_electrodes(self,ax,hemi:str,bipolar:bool,data,color:dict|tuple=(0,0,0,1)):
            
            """Plot effect on electrodes."""
            return ax
      
      def example_registration_figure(self,hemi:str,bipolar:bool)->None:
            """Example registration figure."""
            import pyvista as pv
            from modules.helper_functions import polydata_from_gifti
            
            pt_midthickness = GiftiImage.from_filename(self.data_root / 'pt-space/surf' / f'{hemi}.midthickness.surf.gii')
            pt_sphere = GiftiImage.from_filename(self.data_root / 'pt-space/surf' / f'{hemi}.{self.atlas.pt_sphere_name}')
            
            testpoint = pt_midthickness.agg_data('pointset')[np.random.randint(pt_midthickness.agg_data('pointset').shape[0])]
            vert = pdist2(np.array([testpoint]),pt_midthickness.agg_data('pointset'),num_mins=1)[1][0]
            pt_sphere_point = pt_sphere.agg_data('pointset')[vert]
            
            fs_LR_midthickness= GiftiImage.from_filename(self.data_root / 'pt-space' / f'fs_LR.{self.atlas.name}' / 'surf' / f'{self.atlas.name}.{hemi}.midthickness.surf.gii')
            fs_LR_sphere = GiftiImage.from_filename(self.data_root / 'pt-space' / f'fs_LR.{self.atlas.name}' / 'surf' / f'{self.atlas.name}.{hemi}.sphere.surf.gii')
            fname ='topology-example.png' 
            def screenshot1(vol: pv.Plotter)->None:
                  """Screenshot1."""
                  vol.screenshot(fname,transparent_background=True)
            
            vol = pv.Plotter(shape=(2,3))
            vol.link_views()
            vol.subplot(0,0)
            vol.add_mesh(polydata_from_gifti(pt_midthickness),color='red',opacity=0.25)
            vol.add_mesh(pv.PolyData(testpoint.reshape(1,3)),color='black',point_size=20,render_points_as_spheres=True)
            vol.subplot(1,0)
            vol.add_mesh(polydata_from_gifti(pt_sphere),color='red',opacity=1,show_edges=True)
            vol.add_mesh(pv.PolyData(pt_sphere_point.reshape(1,3)),color='black',point_size=20,render_points_as_spheres=True)
            vol.subplot(0,1)
            vol.add_mesh(polydata_from_gifti(pt_midthickness),color='red',opacity=0.25)
            vol.add_mesh(polydata_from_gifti(fs_LR_midthickness),color='blue',opacity=0.25)
            vol.add_mesh(pv.PolyData(testpoint.reshape(1,3)),color='black',point_size=20,render_points_as_spheres=True)
            vol.subplot(1,1)
            vol.add_mesh(polydata_from_gifti(pt_sphere)    ,color='red',opacity=0.5 )
            vol.add_mesh(polydata_from_gifti(fs_LR_sphere) ,color='blue',opacity=0.5 )
            vol.add_mesh(pv.PolyData(pt_sphere_point.reshape(1,3)),color='black',point_size=20,render_points_as_spheres=True)
            
            vol.subplot(0,2)
            vol.add_mesh(polydata_from_gifti(fs_LR_midthickness),color='blue',opacity=0.25)
            vol.add_mesh(pv.PolyData(testpoint.reshape(1,3)),color='black',point_size=20,render_points_as_spheres=True)
            vol.subplot(1,2)
            vol.add_mesh(polydata_from_gifti(fs_LR_sphere) ,color='blue',opacity=1,show_edges=True)
            vol.add_mesh(pv.PolyData(pt_sphere_point.reshape(1,3)),color='black',point_size=20,render_points_as_spheres=True)
            vol.show(before_close_callback=screenshot1)
      
      @property
      def lh_plot_surf(self)->Path:
            return self._lh_plot_surf
      @lh_plot_surf.setter
      def lh_plot_surf(self,fp: Path)->None:
            self._lh_plot_surf = fp
      @property
      def rh_plot_surf(self)->Path:
            return self._rh_plot_surf
      @rh_plot_surf.setter
      def rh_plot_surf(self,fp: Path)->None:
            self._rh_plot_surf = fp

      def set_plotter_surfaces(self,surfs:dict[str,Path])->None:
            for i in ['lh','rh']:
                  attr = f'{i}_plot_surf'
                  self.__setattr__(attr,surfs[i])
      def set_default_views(self)->None:
            views = (
            "Isometric Right",
            "Isometric Left",
            "Coronal",
            "Left",
            "Right",
            "Axial",
            )
            views = [i.lower() for i in views]
            elevations = (0, 0, -37.5, -30, -30, 60)
            azimuths = (0, 90, 45, 135, -45, 45)
            self.view_angles = {i.lower(): [e, a] for i, e, a in zip(views, elevations, azimuths)} 
      
      def generic_surface_plot(self,hemi:str,ax:pv.Plotter=None,scalars:Optional[np.ndarray]=None,show_scalar_bar:bool=False)->pv.Plotter:
            """Render a cortical surface mesh with scalar shading."""
            hemi = hemi.lower()
            if hemi in ['l', 'r']:
                  hemi = self._side2hemi(hemi)
            elif hemi not in ['lh', 'rh']:
                  raise ValueError("hemi must be one of 'L', 'R', 'lh', or 'rh'.")
            
            # if isinstance(surface_type,Path):
            #       fp = surface_type
            # else:
            #       if not surface_type.endswith('.surf.gii'):
            #             surface_type = f'{surface_type}.surf.gii'

            #       fp = self.midthickness_template_path.parent / self.midthickness_template_path.name.replace('[hemi]', hemi)
            #       fp = fp.parent / fp.name.replace('midthickness.surf.gii', surface_type)
            if hemi =='lh': fp = self.lh_plot_surf
            else: fp =self.rh_plot_surf
            
            surface = load_nii_file(fp)
            verts = surface.agg_data('pointset')
            faces = flattenCells(surface.agg_data('triangle'))
            mesh = pv.PolyData(verts, faces)
            if scalars is None:
                  sulcal_fp = self.sulcal_depth_map_template_path_fsLR.parent / self.sulcal_depth_map_template_path_fsLR.name.replace('[hemi]', hemi)
                  scalars = load_nii_file(sulcal_fp).agg_data()
            if len(scalars) != len(verts):
                  raise ValueError(f"scalar length ({len(scalars)}) must match number of vertices ({len(verts)}).")

            if ax is None:
                  ax = pv.Plotter()
            ax.add_mesh(mesh, scalars=scalars, cmap='gray_r', show_scalar_bar=show_scalar_bar)
            return ax
      
      def surfaceplot_additional_ROI(self,key: Hashable, hemi: str, ax:Optional[pv.Plotter]=None, showLegend:bool=False, roi_opacity:float=1.0, outline:bool=False)->pv.Plotter:
            """Overlay ROI colors on a surface plotter."""
            hemi = hemi.lower()
            if hemi in ['l', 'r']:
                  hemi = self._side2hemi(hemi)
            elif hemi not in ['lh', 'rh']:
                  raise ValueError("hemi must be one of 'L', 'R', 'lh', or 'rh'.")

            if ax is None:
                  ax = self.generic_surface_plot(hemi=hemi)
            if hemi not in self.additional_ROIs[key]:
                  return None
            data: np.ndarray = self.additional_ROIs[key][hemi]
            vmap = self.additional_ROIs[key]['vmap']
            cmap_dict = self.additional_ROIs[key]['cmap']
            
            if hemi =='lh': fp = self.lh_plot_surf
            else: fp =self.rh_plot_surf
                        
            surface = load_nii_file(fp)
            verts = surface.agg_data('pointset')
            faces = surface.agg_data('triangle')
            if len(data) != len(verts):
                  raise ValueError(f"ROI data length ({len(data)}) must match number of vertices ({len(verts)}).")
            legend_entries = []
            used_labels = set()
            unique = np.unique(data)
            unique = unique[unique != 0]
            roi_faces_list = []
            roi_face_rgba = []
            outline_overlays: list[tuple[pv.PolyData, tuple[float, float, float]]] = []
            for v in unique:
                  mask = (data == v)
                  tri_mask = np.all(mask[faces], axis=1)
                  if not np.any(tri_mask):
                        continue
                  label = vmap[v]
                  color = cmap_dict[label]
                  alpha = color[3] if len(color) > 3 else 1.0
                  if alpha <= 0:
                        continue
                  roi_faces = faces[tri_mask]
                  rgb = np.asarray(color[0:3], dtype=float)
                  if rgb.max() <= 1.0:
                        rgb = np.clip(rgb * 255.0, 0, 255)
                  a = int(np.clip(alpha * roi_opacity, 0.0, 1.0) * 255.0)
                  rgba_u8 = np.array([int(rgb[0]), int(rgb[1]), int(rgb[2]), a], dtype=np.uint8)
                  roi_faces_list.append(roi_faces)
                  roi_face_rgba.extend([rgba_u8] * len(roi_faces))

                  if outline:
                        edges = np.vstack([roi_faces[:, [0, 1]], roi_faces[:, [1, 2]], roi_faces[:, [2, 0]]])
                        edges_sorted = np.sort(edges, axis=1)
                        unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
                        boundary_edges = unique_edges[counts == 1]
                        if boundary_edges.shape[0] > 0:
                              line_cells = np.hstack(
                                    [np.full((boundary_edges.shape[0], 1), 2, dtype=np.int64), boundary_edges.astype(np.int64)]
                              ).ravel()
                              boundary_mesh = pv.PolyData(verts, lines=line_cells)
                              outline_overlays.append(
                                    (
                                          boundary_mesh,
                                          tuple(np.asarray(color[0:3], dtype=float))
                                    )
                              )
                  if showLegend and label not in used_labels:
                        legend_entries.append([str(label), tuple(np.asarray(color[0:3], dtype=float))])
                        used_labels.add(label)

            if len(roi_faces_list) > 0 and not outline:
                  roi_faces_all = np.vstack(roi_faces_list)
                  roi_face_cells = flattenCells(roi_faces_all)
                  roi_mesh = pv.PolyData(verts, roi_face_cells)
                  roi_mesh.cell_data['roi_rgba'] = np.vstack(roi_face_rgba)
                  ax.add_mesh(
                        roi_mesh,
                        scalars='roi_rgba',
                        rgb=True,
                        show_scalar_bar=False,
                        style='surface'
                  )
            if outline:
                  for boundary_mesh, rgb in outline_overlays:
                        ax.add_mesh(
                              boundary_mesh,
                              color=rgb,
                              line_width=2.0,
                              render_lines_as_tubes=True,
                              opacity=1.0
                        )

            if showLegend and len(legend_entries) > 0:
                  ax.add_legend(legend_entries)
            return ax
      
      def launch_roi_multiview_gui(self, roi_keys: Hashable|List[Hashable], surf_name:str='midthickness', roi_opacity:float=1.0, block:bool=True):
            """
            Launch ROI multiview scaffold from data_viewing.py.
            Current version renders flatmap ROI overlays in the top row and keeps
            surface panes as placeholders for iterative development.
            """
            try:
                  from PyBrain.modules.data_viewing import launch_roi_multiview_roi
            except ModuleNotFoundError or ImportError:
                  try:
                        from .data_viewing import launch_roi_multiview_roi  # type: ignore
                  except ModuleNotFoundError or ImportError:
                        from data_viewing import launch_roi_multiview_roi  # type: ignore
            return launch_roi_multiview_roi(
                  atlas_obj=self,
                  roi_keys=roi_keys,
                  block=block,
                  title="ROI Multi-View",
                  surf_name=surf_name,
                  roi_opacity=roi_opacity
            )
      
      
      def load_flat_electrodes(self,side: str, bipolar: bool)-> dict:
            """Load flat electrodes."""
            if len(side)>1: hemi = side.lower(); side=side[0].upper()
            elif side.upper() == 'L': side='L'; hemi = 'lh'
            else: side='R'; hemi = 'rh'
            try:
                  if bipolar: fname = self.electrode_path.parent.parent / 'electrodes_bipolar' / self.electrode_path.name.replace('[hemi]',hemi)
                  else: fname = self.electrode_path.parent / self.electrode_path.name.replace('[hemi]',hemi)
                  with open(fname,'r') as fp: 
                        electrodes:dict = json.load(fp)
                  return electrodes
            except AttributeError:
                  return {}

      def export_additional_rois_to_func_gii(
            self,
            out_dir: str | Path,
            roi_keys: Optional[List[Hashable]] = None
      ) -> List[Path]:
            """Export additional ROI overlays to `.func.gii` files."""
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if roi_keys is None:
                  keys_to_export: List[Hashable] = list(self.additional_ROIs.keys())
            else:
                  keys_to_export = list(roi_keys)

            written: List[Path] = []
            hemis = ['lh', 'rh']
            for key in keys_to_export:
                  if key not in self.additional_ROIs:
                        continue
                  roi_entry = self.additional_ROIs[key]
                  if not isinstance(roi_entry, dict):
                        continue
                  for hemi in hemis:
                        if hemi not in roi_entry:
                              continue
                        data = np.asarray(roi_entry[hemi], dtype=np.float32).reshape(-1)
                        arr = nibabel.gifti.GiftiDataArray(
                              data=data,
                              intent='NIFTI_INTENT_SHAPE'
                        )
                        img = nibabel.gifti.GiftiImage(darrays=[arr])
                        outfile = out_dir / f"{self.atlas.name}.{hemi}.{str(key)}.func.gii"
                        nibabel.save(img, outfile)
                        written.append(outfile)

            return written

      def export_electrodes_to_func_gii(
            self,
            out_dir: str | Path,
            include_bipolar: bool = False,
            electrode_value: float = 1.0
      ) -> List[Path]:
            """Export electrode vertex overlays to `.func.gii` files."""
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if not hasattr(self, 'electrode_path'):
                  return []

            written: List[Path] = []
            hemis = ['lh', 'rh']
            bipolar_flags = [False, True] if include_bipolar else [False]

            for bipolar in bipolar_flags:
                  suffix = 'electrodes_bipolar' if bipolar else 'electrodes'
                  for hemi in hemis:
                        elec = self.load_flat_electrodes(hemi, bipolar=bipolar)
                        surf_fp = self.midthickness_template_path_fsLR.parent / self.midthickness_template_path_fsLR.name.replace('[hemi]', hemi)
                        n_verts = load_nii_file(surf_fp).agg_data('pointset').shape[0]
                        values = np.zeros(n_verts, dtype=np.float32)

                        for rec in elec.values():
                              try:
                                    v = int(rec['vert'])
                              except (KeyError, TypeError, ValueError):
                                    continue
                              if 0 <= v < n_verts:
                                    values[v] = np.float32(electrode_value)

                        arr = nibabel.gifti.GiftiDataArray(
                              data=values,
                              intent='NIFTI_INTENT_SHAPE'
                        )
                        img = nibabel.gifti.GiftiImage(darrays=[arr])
                        outfile = out_dir / f"{self.atlas.name}.{hemi}.{suffix}.func.gii"
                        nibabel.save(img, outfile)
                        written.append(outfile)

            return written

      def export_surface_overlays_to_func_gii(
            self,
            out_dir: str | Path,
            roi_keys: Optional[List[Hashable]] = None,
            include_electrodes: bool = True,
            include_bipolar: bool = False,
            electrode_value: float = 1.0
      ) -> List[Path]:
            """Backward-compatible wrapper for ROI/electrode func.gii exports."""
            written = self.export_additional_rois_to_func_gii(out_dir=out_dir, roi_keys=roi_keys)
            if include_electrodes:
                  written.extend(
                        self.export_electrodes_to_func_gii(
                              out_dir=out_dir,
                              include_bipolar=include_bipolar,
                              electrode_value=electrode_value
                        )
                  )
            return written

      def ROI_spatial_PCA(self,hemi:str,parcels:Hashable|List[Hashable],projection_type:str='pt',visualize_axes:bool=True)->tuple:
            from modules.statistics_and_math import run_PCA
            """
            spatial_PCA: Takes a parcel (or parcels) from a given brain annotation, and compute the first two principle axes. Enables spatial warping along primary axes of the geometry, can then be used for normalization and group projection along a shared axis.

            Args:
                  hemi (str): hemisphere to target
                  parcels (Hashable | List[Hashable]): key or list of keys from anatomical parcels. if key is not found, additional ROIs will be checked. If all keys are not found, returns a KeyError.
                  projection_type (Hashable): key to project on pt brain or fs_LR brain topology. defaults to pt
            Returns:
                  tuple: PCA componets, variance explained and transformed data. 
            """
            hemi = hemi.lower()
            if hemi in ['l', 'r']:
                  hemi = self._side2hemi(hemi)
            elif hemi not in ['lh', 'rh']:
                  raise ValueError("hemi must be one of: 'L', 'R', 'lh', or 'rh'.")

            if isinstance(parcels,Hashable):
                  parcels = [parcels]
            fp = self.midthickness_template_path_pt
            surf_fp = fp.parent / fp.name.replace('[hemi]',hemi)
            
            fp = self.annot_file_template_path_pt
            annot_fp = fp.parent / fp.name.replace('[hemi]',hemi)
            
            
            
            surface = load_nii_file(surf_fp)
            verts = surface.agg_data('pointset')
            # faces = surface.agg_data('triangle')
            annot_file = load_nii_file(annot_fp)
            annot_data = np.asarray(annot_file.agg_data()).copy()
            gifti_labels = annot_file.labeltable
            cmap,region_map = annotations_from_gifti_labels(gifti_labels)
            inverse_region_map = {k:i for i,k in region_map.items()}
            
            output_verts = np.empty([0,3])
            for p in parcels:
                  parcelID = inverse_region_map[p]
                  surf_mask = annot_data==parcelID
                  target_verts = verts[surf_mask,:]
                  output_verts = np.vstack((output_verts,target_verts))
            model,transformed_dat,fig = run_PCA(output_verts,n_components=3,visualize_axes=visualize_axes)
            ROI_str = ', '.join(parcels)
            if visualize_axes:
                  fig.suptitle(f'{hemi}, {ROI_str}')
            
            
            return model,transformed_dat,fig,output_verts
      
      def electrode_RAS_to_PC_space(self,PCA_model, electrodes: dict)->dict:
            PCs = PCA_model.components_
            keys = list(electrodes.keys())
            coords = np.asarray(list(electrodes.values()))
            coords_transformed = (coords-PCA_model.mean_) @ PCs.T
            
            return keys,coords_transformed

      def orient_PC1_positive(self, reference_geometry: np.ndarray, *other_geometries: np.ndarray) -> tuple[float, float, np.ndarray, tuple[np.ndarray, ...]]:
            """Orient PC1 from the reference geometry, then apply the same flip and shift to all other geometries."""
            reference_out = np.array(reference_geometry, copy=True)
            others_out = tuple(np.array(geometry, copy=True) for geometry in other_geometries)

            sign = 1.0
            reference_min = float(np.min(reference_out[:, 0]))
            reference_max = float(np.max(reference_out[:, 0]))
            if abs(reference_min) > abs(reference_max):
                  sign = -1.0
                  reference_out[:, 0] *= sign
                  others_out = tuple(
                        np.column_stack((geometry[:, 0] * sign, geometry[:, 1:]))
                        for geometry in others_out
                  )

            offset = float(np.min(reference_out[:, 0]))
            reference_out[:, 0] -= offset
            others_out = tuple(
                  np.column_stack((geometry[:, 0] - offset, geometry[:, 1:]))
                  for geometry in others_out
            )

            return sign, offset, reference_out, others_out
      
      def apply_PC_normalization(self,transformed_brain,transformed_electrodes)->tuple:
            """Normalize the space based on the maximum value of the tissue in PC space, which should be along the extent of PC 1."""
            normalization_values = np.max(np.abs(transformed_brain[:,0]))
            brain_norm = np.divide(transformed_brain,normalization_values)
            electrodes_norm = np.divide(transformed_electrodes,normalization_values)         
            return normalization_values, brain_norm, electrodes_norm
      
      def export_PC_projected_data(self)->None:
            pass
            
      def full_RAS_and_PC_electrodes_and_brain_processing(self,hemi:str,parcels:Hashable|List[Hashable],electrodes:dict, projection_type:str='pt',visualize_axes:bool=True):
            
            PCA_model, transformed_cortex, _, original_cortex = self.ROI_spatial_PCA(hemi,parcels,projection_type,False)
            electrode_channels,transformed_e_coords = self.electrode_RAS_to_PC_space(PCA_model,electrodes)
            _, _, transformed_cortex, oriented_geometries = self.orient_PC1_positive(transformed_cortex, transformed_e_coords)
            transformed_e_coords = oriented_geometries[0]
            norm_vals, norm_brain, norm_e_coords = self.apply_PC_normalization(transformed_cortex,transformed_e_coords)
            
            fig = plt.figure()
            axRAS = fig.add_subplot(131, projection='3d')
            axPC = fig.add_subplot(132, projection='3d')
            axNorm = fig.add_subplot(133, projection='3d')
            axs = [axRAS,axPC,axNorm]
            axRAS.scatter(original_cortex[:,0],original_cortex[:,1],original_cortex[:,2],alpha=0.2,s=1)
            axPC.scatter(transformed_cortex[:,0],transformed_cortex[:,1],transformed_cortex[:,2],alpha=0.2,s=1)
            axNorm.scatter(norm_brain[:,0],norm_brain[:,1],norm_brain[:,2],alpha=0.2,s=1)
            geometries = [
                  original_cortex,
                  transformed_cortex,
                  norm_brain
            ]
            axis_labels = [
                  ('R', 'A', 'S'),
                  ('PC1', 'PC2', 'PC3'),
                  ('nPC1', 'nPC2', 'nPC3')
            ]
            for ax, geometry, labels in zip(axs, geometries, axis_labels):
                  center = np.mean(geometry, axis=0)
                  ranges = np.ptp(geometry, axis=0)
                  half_range = float(np.max(ranges) / 2.0)
                  if half_range == 0.0:
                        half_range = 1.0
                  ax.set_xlim(center[0] - half_range, center[0] + half_range)
                  ax.set_ylim(center[1] - half_range, center[1] + half_range)
                  ax.set_zlim(center[2] - half_range, center[2] + half_range)
                  ax.set_box_aspect((1.0, 1.0, 1.0))
                  ax.set_xlabel(labels[0])
                  ax.set_ylabel(labels[1])
                  ax.set_zlabel(labels[2])
                  ax.grid(False)
            # for idx,i in enumerate(electrode_channels):
                  # pass
                  
                  
                  
                  
                  
            fig.tight_layout()
            
            return fig
      
      
def unit_rgb(c)-> Tuple[float]:
      """Unit rgb."""
      c = np.asarray(c, float)
      return tuple(c/255.0) if c.max() > 1.0 else tuple(c)

def replace_labels(labels,label_IDs,targetLabels)->np.ndarray:
      """Replace labels."""
      return np.zeros([1,1])

def label_patches(ax:Axes,verts_xy,faces,labels,cmap,outline=True,opacity: float=1,legend:Optional[dict]=None)->Axes:
      """Label patches."""
      from matplotlib.colors import ListedColormap
      for target_ID in np.unique(labels):
            if target_ID <0:
                  continue
            if legend is not None: legend_entry = legend[target_ID]
            else: legend_entry = '_'
            locs = np.flatnonzero(labels == target_ID)
            if locs.size > 0:
                  verts = verts_xy[locs,:]
                  vert_mapping = {g:i for i,g in enumerate(locs)}
                  in_label = np.isin(faces, locs)
                  keep = in_label.all(axis=1)
                  tris_g = faces[keep]
                  if tris_g.any():
                        # remap triangle vertex indices from global->local
                        tris_l = np.vectorize(vert_mapping.__getitem__)(tris_g)

                        tri = mtri.Triangulation(verts[:, 0], verts[:, 1], tris_l)
                        
                        # z = np.tile(unit_rgb(cmap[target_ID]),(len(verts),1))
                        z = np.ones(verts.shape[0])
                        rgb = unit_rgb(cmap[target_ID])
                        if not outline:
                              rgb = list(rgb)
                              rgb[-1] = opacity
                              aaa = ListedColormap(rgb)
                              # pc = ax.tripcolor(tri, np.zeros(verts.shape[0]),cmap=aaa)
                              pc = ax.tripcolor(tri, z,cmap=aaa,label=legend_entry,zorder=2)
                              # pc.set_array(None)
                              # pc.set_facecolor([0,0,0,0])
                              # pc.set_edgecolor(rgb)
                        else:
                              V = verts                                      # tri vertex array (n,2)
                              T = tri.triangles.astype(int)                  # (m,3)

                              edges = np.vstack([T[:,[0,1]], T[:,[1,2]], T[:,[2,0]]])
                              edges_sorted = np.sort(edges, axis=1)
                              unique, inv, counts = np.unique(edges_sorted, axis=0, return_inverse=True, return_counts=True)
                              outer_mask = counts[inv] == 1                  # edges that appear once
                              outer_edges = edges[outer_mask]                # keep original orientation

                              segs = np.stack([V[outer_edges[:,0]], V[outer_edges[:,1]]], axis=1)
                              lc = LineCollection(segs, colors=[rgb], linewidths=2.5,label=legend_entry,zorder=2)# type: ignore
                              ax.add_collection(lc)
      if legend is not None:
            ax.legend(bbox_to_anchor=(1.051, 1.025))
      return ax

def adjust_lightness (color, amount=1.2)-> tuple:
      """adjusts the lightness of 1 color"""
      import matplotlib.colors as mc
      import colorsys
      try:
            c = mc.cnames[color]
      except:
            c = color
      c = colorsys.rgb_to_hls(*mc.to_rgb(c))
      return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])



def face_majority_labels(vertex_labels, faces_tri)->np.ndarray:
      """Label per face = majority vote of its 3 vertex labels (>=0), else -1."""
      L = vertex_labels
      fL = np.full(len(faces_tri), -1, int)
      for fi, (i,j,k) in enumerate(faces_tri):
            tri = [L[i], L[j], L[k]]
            tri = [t for t in tri if t >= 0]
            if not tri: 
                  continue
            a = tri.count(tri[0])
            # majority in {tri[0], other}
            if len(tri) == 1:
                  fL[fi] = tri[0]
            else:
                  other = next(t for t in tri if t != tri[0]) if any(t!=tri[0] for t in tri) else tri[0]
                  b = tri.count(other)
                  fL[fi] = tri[0] if a >= b else other
      return fL

def edge_face_adjacency(faces_tri)-> dict:
      """Map undirected edge -> incident face indices."""
      adj = {}
      for fi, (i,j,k) in enumerate(faces_tri):
            for a,b in ((i,j),(j,k),(k,i)):
                  e = (a,b) if a<b else (b,a)
                  adj.setdefault(e, []).append(fi)
      return adj

def draw_outlines_per_parcel(ax,lab,verts,edges,color_map, lw=0.8)->Axes:
      """Draw outlines per parcel."""
      lab0 = lab[edges[:,0]]
      lab1 = lab[edges[:,1]]
      valid = (lab0 >=0) & (lab1 >=0)
      for L, rgb in color_map.items():
            if L not in lab: 
                  continue
            m = valid & ((lab0 == L) ^ (lab1 == L))   # edges with exactly one endpoint in parcel L
            anyEdges = m.any()
            if not anyEdges:
                  continue
            segs = np.stack([verts[edges[m,0]], verts[edges[m,1]]], axis=1)   # (k,2,2)
            lc = LineCollection(segs, colors=[unit_rgb(rgb)], # type: ignore
                  linewidths=lw-lw*-0.25, antialiased=False, zorder=9)
            lc = LineCollection(segs, colors=[unit_rgb(rgb)], # pyright: ignore[reportArgumentType]
                  linewidths=lw, antialiased=False, zorder=10)
            lc.set_joinstyle('miter')
            ax.add_collection(lc)
      return ax 

def electrode_name_match(electrodes:dict,target_names: List[Hashable]=None,delim:str='-'):
      # TODO: make this more robust, the initial attempt at fuzzy matching did not work very well.
      # from rapidfuzz import fuzz,process
      """Electrode name match."""
      name_mapping = {i:''.join([i.split(delim)[0],i.split('_')[-1]]) for i in electrodes}
      # target_names = [i.replace(' ','') for i in target_names]
      # stringDict = {i:process.extractOne(i,target_names)[0] for i in electrodes_short}
      output = {name_mapping[i]:j for i,j in electrodes.items()}
      return output, name_mapping
      
class groupAtlas(projectAtlas):
      def __init__(self,subjects: List[str],template_dir:Path,atlas: Optional[Atlas]=None)->None:
            """  init  ."""
            self.subjects = list(np.unique(subjects))
            self.root: Path = template_dir.parent
            self.atlas = atlas
            self.data_root: Path = self.root/'gifti'
            self._electrode_library = self._load_group_electrodes()
            self.init_fsLR_template_paths()
      
      @property
      def electrode_library(self)->pd.DataFrame:
            """
            electrode_library all group projected electrodes available for the instance of GroupAtlas.
            electrode_library has the following columns
                  - channel: label from single subjet.
                  - vert: nearest vertex to electrode  in subject RAS space.
                  - dist: distance to the vert from original electrode in subject RAS space.
                  - label: class label of the electrode.
                  - region: brain region associated with electrode in original subject RAS space.
                  - flatcoords: flatmap coordinates on subject specific flatmap.
                  - subject: subject electrode was implanted in.
                  - hemi: hemisphere electrode is assigned to. 
            """
            return self._electrode_library

      def export_electrode_summary(self, export_dir: str | Path, print2Latex: bool = False) -> Path:
            """
            Export a CSV summary of electrode and trajectory counts by subject and hemisphere.

            Args:
                  export_dir (str | Path): Directory where the summary CSV will be written.
                  print2Latex (bool, optional): Print a LaTeX-formatted table to stdout after saving the CSV.

            Returns:
                  Path: Path to the exported CSV file.
            """
            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            import re
            def parse_trajectory(channel: str) -> str:
                  
                  root = channel.replace('-b-','=')
                  # m = re.search(r'[A-Za-z](?!.*[A-Za-z])', root)
                  m = max((i for i, c in enumerate(root) if c.isalpha()), default=-1)
                  return root[0:m+1]

            electrode_data = self.electrode_library.reset_index().copy()
            electrode_data['trajectory'] = electrode_data['channel'].map(parse_trajectory)

            counts = (
                  electrode_data
                  .groupby(['subject', 'hemi'])
                  .agg(
                        electrode_count=('channel', 'size'),
                        trajectory_count=('trajectory', 'nunique'),
                  )
                  .unstack(fill_value=0)
            )
            counts = counts.reindex(columns=pd.MultiIndex.from_product(
                  [['electrode_count', 'trajectory_count'], ['lh', 'rh']]
            ), fill_value=0)
            counts.columns = [
                  'lh_electrode_count',
                  'rh_electrode_count',
                  'lh_trajectory_count',
                  'rh_trajectory_count',
            ]
            counts = counts.reset_index()

            total_row = pd.DataFrame(
                  [{
                        'subject': 'total',
                        'lh_electrode_count': int(counts['lh_electrode_count'].sum()),
                        'rh_electrode_count': int(counts['rh_electrode_count'].sum()),
                        'lh_trajectory_count': int(counts['lh_trajectory_count'].sum()),
                        'rh_trajectory_count': int(counts['rh_trajectory_count'].sum()),
                  }]
            )

            avg_row = pd.DataFrame(
                  [{
                        'subject': 'mean',
                        'lh_electrode_count':np.average((counts['lh_electrode_count'])),
                        'rh_electrode_count':np.average((counts['rh_electrode_count'])),
                        'lh_trajectory_count': np.average((counts['lh_trajectory_count'])),
                        'rh_trajectory_count': np.average((counts['rh_trajectory_count'])),
                  }]
            )
            
            summary = pd.concat([counts, total_row, avg_row], ignore_index=True)
            

            outfile = out_dir / 'electrode_summary.csv'
            summary.to_csv(outfile, index=False)
            if print2Latex:
                  latex_rows: list[str] = []
                  for idx, row in summary.iterrows():
                        if idx == len(summary) - 1:
                              latex_rows.append("        \\midrule")
                        if row['subject']!= 'mean':
                              latex_rows.append(
                                    "        "
                                    f"{row['subject']} & "
                                    f"{int(row['lh_electrode_count'])} & "
                                    f"{int(row['rh_electrode_count'])} & "
                                    f"{int(row['lh_trajectory_count'])} & "
                                    f"{int(row['rh_trajectory_count'])} \\\\"
                              )
                        else:
                              latex_rows.append(
                                    "        "
                                    f"{row['subject']} & "
                                    f"{np.round(row['lh_electrode_count'],2)} & "
                                    f"{np.round(row['rh_electrode_count'],2)} & "
                                    f"{np.round(row['lh_trajectory_count'],2)} & "
                                    f"{np.round(row['rh_trajectory_count'],2)} \\\\"
                              )
                  latex_table = "\n".join([
                        "\\begin{table}",
                        "    \\centering",
                        "    \\begin{tabular}{lrrrr}",
                        "        \\toprule",
                        "        Subject & LH electrodes & RH electrodes & LH trajectories & RH trajectories \\\\",
                        "        \\midrule",
                        *latex_rows,
                        "        \\bottomrule",
                        "    \\end{tabular}",
                        "    \\label{tab:coverage_sampling}",
                        "    \\caption{Electrode coverage data per participant included in the aggregate group map. Electrodes refer to individual recording contacts, while trajectories refer to implanted shanks containing multiple individual electrodes. LH: Left Hemisphere, RH: Right Hemisphere.}",
                        "\\end{table}",
                  ])
                  print(latex_table)
            return outfile
      
      
      def project_electrodes_to_single_hemi(self,target_hemi: str,overrite_data:bool=False)->pd.DataFrame:
            sphere,hemi,tag = self._checkHemi(target_hemi)
            side = self._hemi2side(hemi)
            if hemi == 'lh':
                  proj_hemi = 'rh'
            else: proj_hemi = 'lh'
            proj_side = self._hemi2side(proj_hemi)
            df = self.electrode_library.copy()
            df['original_hemi'] = self.electrode_library['hemi']
            df['hemi'] = target_hemi
            df = df.drop('flatcoords',axis=1)
            if overrite_data: self._electrode_library = df
            
            return df
      
      
      def get_electrode_bbox(self, electrodes:pd.DataFrame)-> tuple:
            """
            get_electrode_bbox extract x and y bounds of a set of electrodes passed to the function for dynamic image alignment.

            Args:
                  electrodes (pd.DataFrame): dataframe of electrode locations. Included to slice the electrodes produced by self.electrode_library to zoom in on a specific ROI on the map. 

            Returns:
                  tuple: returns tuple of bounds in format (xmin,xmax,ymin,ymax)
            """
            coords = np.stack(electrodes['flatcoords'].values)
            coords = coords[:,0:-1]
            xmax,ymax = np.max(coords,axis=0)
            xmin,ymin = np.min(coords,axis=0)
            
            
            return xmin,xmax,ymin,ymax
      
      
      def _load_group_electrodes(self,bipolar:bool=True)->pd.DataFrame:
            """
            load_group_electrodes creates dataframe aggregating electrodes projected to surfaces across patient cohort

            Args:
                  bipolar (bool, optional): bipolar re-references contacts. Defaults to True.

            Returns:
                  pd.DataFrame: pandas dataframe of electrodes with
            """
            dfOut = pd.DataFrame()
            for i in self.subjects:
                  fdir=self.root.parent / i / 'gifti' / 'pt-space' / f'fs_LR.{self.atlas.name}'
                  if bipolar: elec_dir = fdir / 'electrodes_bipolar'
                  else: elec_dir = fdir/'electrodes'
                  df = pd.DataFrame()
                  for j in os.listdir(elec_dir):
                        filename = elec_dir/j
                        with open(filename,'r') as fp:
                              temp = json.load(fp)
                        temp2 = {''.join([i.split('_')[0].split('-')[0].replace("'","L"),i.split('_')[-1]]):j for i,j in temp.items()}
                        temp = pd.DataFrame.from_dict(temp2).T
                        temp['subject'] = i
                        temp['hemi'] = j.split('.')[0]
                        df = pd.concat([df,temp])
                  df = df.reset_index().rename({'index':'channel'},axis=1).sort_values('dist').drop_duplicates(subset='channel',keep='first').sort_values('channel')
                  
                  dfOut = pd.concat([dfOut,df]) 
            dfOut = dfOut.set_index(['subject','channel'])
                        
            return dfOut
      
      def euclidean_neighborhood_voting_map(self,data:pd.DataFrame, r:float=5.0, sigma:float=5.0, k_neighbors:int=10, effect_name:Hashable|List[Hashable]=None)->None:
            """Euclidean neighborhood voting map."""
            from scipy.spatial import cKDTree
            """
            assign class labels to local surface voxels my computing a neighborhood-wide vote based on proximity to sparse electrodes, weighted by electrodes distance and (optional) effect magnitude

            Args:
                  data (pd.DataFrame): dataframe of electrodes with vertex ID, hemisphere, class label and optional effects. See self.electrode_library for details on outputs
                  r (float, optional): _description_. Defaults to 5.0.
                  sigma (float, optional): _description_. Defaults to 5.0.
                  k_neighbors (int, optional): _description_. Defaults to 10.
                  effect_name (Hashable|List[Hashable], optional): effect name(s) to weight votes by, must be contained within column names of data. Defaults to None.
            """
            if effect_name is not None:
                  pass
            def build_sparse_distances(tree:cKDTree, coords:np.ndarray,r:float)->list:
                  
                  """Build sparse distances."""
                  pass
            voting_matrix = {}
            voting_results = {}
            classes = np.unique(data['class'])
            class_id_map = {i:idx+1 for idx,i in enumerate(classes)}
            class_id_map['non-indexed'] = 0 
            class_ids = np.array(list(class_id_map.values()),dtype=int)
            for hemi in ['lh','rh']:
                  fp = self.midthickness_template_path_fsLR.parent/self.midthickness_template_path_fsLR.name.replace('[hemi]',hemi)
                  surf = load_nii_file(fp)
                  mesh_verts, _ = surf.agg_data()

                  distance_weights = np.zeros([len(classes)+1,mesh_verts.shape[0]])
                  electrodes = data.loc[data['hemi']==hemi]
                  e_vert_IDs = np.array(electrodes['vert'].to_numpy(),dtype=int)
                  e_verts = mesh_verts[e_vert_IDs]
                  tree = cKDTree(mesh_verts)
                  target_points = tree.query_ball_point(e_verts,r)
                  e_class = np.array([class_id_map[i] for i in electrodes['class']],dtype=int)
                  for i,e,c in zip(target_points,e_verts,e_class):
                        d = np.linalg.norm(mesh_verts[i] - e, axis=1)
                        k = np.exp(-d**2 / (2 * sigma**2))
                        distance_weights[c,i] += k
                        pass
                  total_support = np.sum(distance_weights,axis=0)
                  winning_class = np.argmax(distance_weights,axis=0)
                  winning_vote_magnitude = np.max(distance_weights,axis=0)
                  voting_matrix[hemi] = distance_weights
                  voting_results[hemi] = winning_class
            
            return voting_matrix, voting_results, {v:k for k,v in class_id_map.items()}
      
      def plot_electrodes(self,ax: Axes,hemi:str,bipolar:bool,subset:Optional[list]=None,color:dict|tuple=(0,0,0,1),size:dict|float=5,useCoords:bool=True,alphas:Optional[list]=None)->Axes:
            """Plot electrodes."""
            electrodes = self.electrode_library
            if subset is not None: 
                  if subset[0]=='regions':
                        electrodes,_ = self.parse_electrodes_from_regions(electrodes,subset[1])
                  if subset[0]=='names':
                        electrodes,_ = self.parse_electrodes_from_names(electrodes,subset[1])
            if isinstance(size,float): 
                  size = {i:size for i in electrodes}
            if np.size(color)<=3: 
                  color = {i:color for i in electrodes}
            if useCoords:
                  if isinstance(electrodes,dict):
                        for k,v in electrodes.items():
                              coords = v['flatcoords']
                              ax.scatter(coords[0],coords[1],color=color[k],s=size[k],zorder=10)
                  else:
                        if alphas is  None:
                              alphas = 0.7
                        coords = np.array(electrodes['flatcoords'].to_list())
                        colors = np.array(color)
                        ax.scatter(
                              x=coords[:,0],y=coords[:,1],c=colors,alpha=alphas
                              )
            else:
                  # TODO: reindex the electrode coordinates based on the vertex assigned and the hemisphere, sanity check, should result in the same locations as the flatmap is already a group map from colin.
                  pass
            
            return ax

if __name__ == '__main__':
      s = 'BJH0'
      nums = [52]
      # if True:
      atlas: Atlas = Atlas.fs_LR_from_fsav('164k')
      atlas.pt_sphere_name = 'sphere.reg.surf.gii' 
      subject = 'fsaverage_wb'
      seg = f"/Users/nkb/Documents/NCAN/patients/{subject}/segmentation"
      fsav = projectAtlas(seg,atlas=atlas,process_fs=False,process_gifti=True)
      aL = fsav.flatmap_plot('L')
      figL = plt.gcf()
      aR = fsav.flatmap_plot('R')
      figR = plt.gcf()
      for n in nums:
            n = str(n)
            subject = ''.join([s,n])
            # subject = 'BJH041'
            print('----------\n')
            print(subject)
            try:
                  seg = f"/Users/nkb/Documents/NCAN/patients/{subject}/segmentation"
                  electrodes_dir = f"/Users/nkb/Documents/NCAN/patients/{subject}/electrodes_clean"
                  if not os.path.exists(electrodes_dir): electrodes_dir = f"/Users/nkb/Documents/NCAN/patients/{subject}/electrodes"
                  bipolar = True
                  flatmap = projectAtlas(seg,atlas=atlas,electrode_dir=electrodes_dir,process_fs=False,process_gifti=True)
                  # flatmap = projectAtlas(seg,atlas=atlas,electrode_dir=electrodes_dir,process_fs=True,process_gifti=True)
                  flatmap.project_bipolar_electrodes(electrodes_dir)
                  hemi = 'L'
            
                  # a = flatmap.flatmap_plot(hemi=hemi,legend=True)
                  try: aL = flatmap.plot_electrodes(aL,hemi,size=10,bipolar=True)
                  except FileNotFoundError or AttributeError: print(f"No electrode projection for {hemi} hemisphere")
                  hemi = 'R'
                  # a = flatmap.flatmap_plot(hemi=hemi,legend=True)
                  try: aR = flatmap.plot_electrodes(aR,hemi,size=10,bipolar=True)
                  except FileNotFoundError or AttributeError: print(f"No electrode projection for {hemi} hemisphere")
                  print(f'finished {subject} successfully')
                  print('\n----------\n')
            
            except Warning as e:
                  print(f'{subject} failed due to {e}')
      
      # fp = Path(f'/Users/nkb/Documents/NCAN/patients/{subject}/gifti/pt-space/fs_LR.32k')
      # patient_out = outdir / 'patients' / subject
      # shutil.copytree(fp,patient_out)
      # figL.savefig(f'L_fsav_{len(nums)}pts',transparent=True)
      # figR.savefig(f'R_fsav_{len(nums)}pts',transparent=True)
      plt.show()
