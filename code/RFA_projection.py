import os
import glob
from pathlib import Path
import numpy as np
import platform
from PyBrain.modules.surface_projection import projectAtlas, Atlas
from PyBrain.modules.spin_nulls import SpinNullModel
from pathlib import Path
import os
import matplotlib.pyplot as plt
import shutil
import matplotlib.colors as mcolors
import pyvista as pv
from plot_SCAN_all_electrodes import ElectrodePlotter
SMN_color = np.asarray([245,139,48])/255
RF_color = mcolors.to_rgb(r'#FFDA03')
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
      dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"
boxpath = Path(os.path.expanduser('~/Library/CloudStorage/Box-Box'))

conte_tag = 'very_inflated'
conte_tag = 'inflated'
conte_L = Path(f"/Users/nkb/Documents/NCAN/atlases/surface_atlases/CONTE69/32k/surfaces/fs_LR.32k.L_hemi/Conte69.L.{conte_tag}.32k_fs_LR.surf.gii")
conte_R = Path(f"/Users/nkb/Documents/NCAN/atlases/surface_atlases/CONTE69/32k/surfaces/fs_LR.32k.R_hemi/Conte69.R.{conte_tag}.32k_fs_LR.surf.gii")


SCANMAP = Path('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/imaging/HCP_Spots_Effectors_CS.dtseries.nii')
SCAN_KEY = 'HCP_SCAN'

conte_surfs = {'lh':conte_L,'rh':conte_R}

SCAN_VMAP = {0:'na', 1.5:'inter',10:'hand',11:'face',17:'foot'}
SCAN_CMAP = {'na':(0,0,0,0),'inter':(158/255,38/255,108/255,1),'hand':(68/255,1,1,1), 'face':(1,142/255,52/255,1),'foot':(32/255,133/255,44/255,1)}

# subjects = ['SLCH014','BJH041']
subjects = ['BJH041','SLCH014']
SCAN_ROI={}
SCAN_ROI['SLCH014']= '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/patients/SLCH014/imaging/NIfTI/SCAN/MSCPI21_SCANmap_allnodes.dtseries.nii'
SCAN_ROI['BJH041']='/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/patients/BJH041/Freesurfer&Workbench/MSC18_SCANROIs.dtseries.nii'
spin_cache_root = userPath / 'Documents' / 'NCAN' / 'patients' / 'spin_tests' / 'RFA_projection'
spin_cache_root.mkdir(parents=True, exist_ok=True)
figure_root = spin_cache_root / 'figures'
figure_root.mkdir(parents=True, exist_ok=True)
num_spin_iters = 10000
regenerate_spins = False
catch_all_threshold = 5
process_flag = False
process_freesurfer = process_flag
process_gifti = process_flag
project_label_flag = False
adjust_affines = False
bipolar = True
atlas_res = '32k'
lookup_keys = ['interop_RFA_FLAIR','LITT']
og_ablation_keys = ['RF_Volume','RF_edema','RF_M12-14','RF_L14-16']

Wp = 2560
Hp = 1440

aspect = Wp/Hp
ang = np.arctan(aspect)
H = np.cos(ang)*27
W = np.sin(ang)*27


def export_blank_flatmaps(
      maps: dict[str, projectAtlas],
      fsav: projectAtlas,
      subjects: list[str],
      outdir: Path,
      figsize: tuple[float, float],
      xlim: list[float],
      ylim: list[float],
) -> None:
      """Export base flatmaps without ROI overlays using the same layout as the main figures."""
      fig_r, ax_r = plt.subplots(1, 3, figsize=figsize)
      fig_l, ax_l = plt.subplots(1, 3, figsize=figsize)

      for idx, subject in enumerate(subjects):
            flatmap = maps[subject]
            flatmap.flatmap_plot('R', ['S_central', 'G_precentral'], legend=False, ax=ax_r[idx])
            flatmap.flatmap_plot('L', ['S_central', 'G_precentral'], legend=False, ax=ax_l[idx])
            ax_r[idx].set_title(subject)
            ax_l[idx].set_title(subject)

      ax_l[-1].set_title('fsav_fsLR32k')
      ax_r[-1].set_title('fsav_fsLR32k')
      fsav.flatmap_plot('L', 'S_central', legend=False, ax=ax_l[-1])
      fsav.flatmap_plot('R', 'S_central', legend=False, ax=ax_r[-1])

      for axes in [ax_l, ax_r]:
            for axis in axes:
                  axis.set_xlim(xlim)
                  axis.set_ylim(ylim)

      for fig in [fig_l, fig_r]:
            fig.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.92)

      fig_l.savefig(outdir / 'RFA_projection_flatmap_left_base.png', dpi=600, bbox_inches='tight')
      fig_r.savefig(outdir / 'RFA_projection_flatmap_right_base.png', dpi=600, bbox_inches='tight')
      plt.close(fig_l)
      plt.close(fig_r)


figR, axR = plt.subplots(1,3,figsize=(W,H))
figL, axL = plt.subplots(1,3,figsize=(W,H))
maps:dict[str,projectAtlas] = {}
spin_jobs: list[dict[str, object]] = []
xlim=[-80,80]
ylim=[-105,170]
for idx, subject in enumerate(subjects):
      print(f'\n-----------\n{subject}')
      aL = axL[idx]
      aR = axR[idx]
      atlas: Atlas = Atlas.fs_LR_from_fsav(atlas_res)
      atlas.pt_sphere_name = 'sphere.reg.surf.gii' 
      seg = f"/Users/nkb/Documents/NCAN/patients/{subject}/segmentation"
      electrodes_dir = f"/Users/nkb/Documents/NCAN/patients/{subject}/electrodes_clean"
      if not os.path.exists(electrodes_dir): electrodes_dir = f"/Users/nkb/Documents/NCAN/patients/{subject}/electrodes"
      label_path = Path(f'/Users/nkb/Documents/NCAN/patients/{subject}/imaging/labels')
      pattern = glob.glob(str(label_path/'*RF*'))+glob.glob(str(label_path/'*LITT*'))
      label_path = [i for i in pattern]
      filetree = boxpath/subject

      flatmap = projectAtlas(seg,atlas=atlas,electrode_dir=electrodes_dir,process_fs=process_freesurfer,process_gifti=process_gifti,buildFileTree=filetree,distanceThreshold=catch_all_threshold, correct_affine=adjust_affines)
      # flatmap.project_bipolar_electrodes(electrodes_dir,thresh=catch_all_threshold)
      lesion_paths = flatmap.project_freesurfer_labels(label_files=label_path,process=project_label_flag)
      target_key = [i for i in lesion_paths for z in lookup_keys if i.find(z)>-1][0]
      
      
      # flatmap.flatmap_plot('L',['S_central'],legend=False,ax=aL)
      flatmap.flatmap_plot('R','S_central',legend=False,ax=aR)
      flatmap.flatmap_plot('L',['S_central','G_precentral'],legend=False,ax=aL)
      flatmap.flatmap_plot('R',['S_central','G_precentral'],legend=False,ax=aR)
      # flatmap.load_cifti_data(SCAN_ROI[subject],'SCAN')
      flatmap.load_cifti_as_ROI(SCAN_ROI[subject],'SCAN')
      flatmap.update_ROI_cmap('SCAN',{'SCAN':(1,1,1)})
      other_lesion_keys = [key for key in lesion_paths if key != target_key and np.any(key.find(z)>-1 for z in og_ablation_keys)]
      if 'RF_edema' in other_lesion_keys: mergeFlag = True
      else: mergeFlag = False
      scan_vmap = flatmap.additional_ROIs['SCAN']['vmap']
      scan_nonzero_values = [value for value in scan_vmap if value != 0]
      scan_subkeys: list[str] = []
      if len(scan_nonzero_values) == 1:
            split_keys = flatmap.split_connected_surface_roi('SCAN', min_component_size=5)
            flatmap.refine_split_surface_roi('SCAN', split_keys, representative_vertex_method='centroid_vertex',max_rois=3)
            scan_subkeys.extend(split_keys)
      else:
            for scan_value in scan_nonzero_values:
                  split_keys = flatmap.split_connected_surface_roi('SCAN', target_value=scan_value, min_component_size=5)
                  flatmap.refine_split_surface_roi('SCAN', split_keys, target_value=scan_value, representative_vertex_method='centroid_vertex',max_rois=3)
                  scan_subkeys.extend(split_keys)
      for hemi,fps in lesion_paths[target_key].items():
            map_path = fps['fs_LR']
            flatmap.load_binary_gifti_as_ROI(map_path,target_key)
            flatmap.update_ROI_cmap(target_key,{target_key:RF_color   })
      for lesion_key in other_lesion_keys:
            for hemi, fps in lesion_paths[lesion_key].items():
                  map_path = fps['fs_LR']
                  flatmap.load_binary_gifti_as_ROI(map_path, lesion_key)
                  flatmap.update_ROI_cmap(lesion_key, {lesion_key: (0.6, 0.6, 0.6)})
      flatmap.flatplot_additional_ROI('SCAN','L',aL,showLegend=False,opacity=1)
      outline_lesion = False
      flatmap.flatplot_additional_ROI(target_key,'L',aL,showLegend=False,opacity=0.5,outline=outline_lesion)
      for lesion_key in other_lesion_keys:
            flatmap.flatplot_additional_ROI(lesion_key,'L',aL,showLegend=False,opacity=0.35,outline=True)
      aL.set_title(subject)
      flatmap.flatplot_additional_ROI('SCAN','R',aR,showLegend=False,opacity=1)
      flatmap.flatplot_additional_ROI(target_key,'R',aR,showLegend=False,opacity=0.5,outline=outline_lesion)
      for lesion_key in other_lesion_keys:
            flatmap.flatplot_additional_ROI(lesion_key,'R',aR,showLegend=False,opacity=0.35,outline=True)
      aR.set_title(subject)
      maps[subject] = flatmap      

      if mergeFlag:
            other_lesion_keys = [flatmap.mergeROIs(other_lesion_keys,'RF_agg',overwrite=True)]
      
      merged_scan_key = flatmap.merge_subroi_keys_to_multilabel(
            subroi_keys=scan_subkeys,
            merged_key=f'SCAN_subrois_{subject}',
            overwrite=True
      )
      merged_scan_data = flatmap.additional_ROIs[merged_scan_key]
      scan_map_labels = {
            hemi: sorted([k for k, value in merged_scan_data['vmap'].items() if k != 0 and hemi in value])
            for hemi in ['lh', 'rh']
      }
      spin_jobs.append({
            'subject': subject,
            'flatmap': flatmap,
            'merged_scan_data': merged_scan_data,
            'scan_map_labels': scan_map_labels,
            'lesion_keys': [target_key, *other_lesion_keys],
      })

atlas.pt_sphere_name = 'sphere.reg.surf.gii' 
subject = 'fsaverage_wb'
seg = userPath / 'Documents'/'NCAN'/'patients'/subject/'segmentation'
fsav =projectAtlas(seg,atlas,process_gifti=False,correct_affine=adjust_affines)
aL = axL[-1]; aL.set_title('fsav_fsLR32k')
aR = axR[-1]; aR.set_title('fsav_fsLR32k')
fsav.flatmap_plot('L','S_central',legend=False,ax=aL)
fsav.flatmap_plot('R','S_central',legend=False,ax=aR)
rois = []
for i,k in maps.items():
      tag = f'_SCAN_{i}'
      fsav.additional_ROIs[tag] = k.additional_ROIs['SCAN']
      rois.append(tag)
      for j in k.additional_ROIs:
            if not np.any([j.find(l)>-1 for l in lookup_keys]):
                  continue
            tag='_'.join([j,i])
            fsav.additional_ROIs[tag] = k.additional_ROIs[j]
            rois.append(tag)
rois=sorted(rois)
print(rois)

fsav.load_cifti_data(SCANMAP,'HCP_SCAN')
fsav.update_additional_ROI_value_map(SCAN_KEY,SCAN_VMAP,SCAN_CMAP)
fsav.flatplot_additional_ROI(SCAN_KEY,'R',ax=aR,outline=True)
fsav.flatplot_additional_ROI(SCAN_KEY,'L',ax=aL,outline=True)


for r in rois:
      fsav.flatplot_additional_ROI(r,'R',ax=aR,opacity=0.4)
      fsav.flatplot_additional_ROI(r,'L',ax=aL,opacity=0.4)
      fsav.flatplot_additional_ROI(r,'R',ax=aR,color_override=(0,0,0),outline=True)
      fsav.flatplot_additional_ROI(r,'L',ax=aL,color_override=(0,0,0),outline=True)
      # fsav.flatplot_additional_ROI(r,'R',ax=aR,opacity=0.7,outline=True)

ax = np.vstack([axL,axR]).ravel()
for a in ax:
      a.set_xlim(xlim)
      a.set_ylim(ylim)
      
for i in plt.get_fignums():
      fig = plt.figure(i)
      plt.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.92)

export_blank_flatmaps(
      maps=maps,
      fsav=fsav,
      subjects=subjects,
      outdir=figure_root,
      figsize=(W, H),
      xlim=xlim,
      ylim=ylim,
)

figL.savefig(figure_root / 'RFA_projection_flatmap_left.png', dpi=300, bbox_inches='tight')
figL.savefig(figure_root / 'RFA_projection_flatmap_left.svg', bbox_inches='tight')
figR.savefig(figure_root / 'RFA_projection_flatmap_right.png', dpi=300, bbox_inches='tight')
figR.savefig(figure_root / 'RFA_projection_flatmap_right.svg', bbox_inches='tight')


# surf_plotter = pv.Plotter(shape=(3,2), window_size=(1800, 700),off_screen=True)
surf_plotter = ElectrodePlotter(shape=(3,2), window_size=(1800, 700),off_screen=True)
for i,sub in enumerate(subjects):
      flatmap = maps[sub]
      flatmap.set_plotter_surfaces(conte_surfs)
      lesion_keys = [key for key in flatmap.additional_ROIs if np.any([key.find(label_key) > -1 for label_key in lookup_keys])]
      if len(lesion_keys) == 0:
            continue
      target_key = lesion_keys[0]
      outline_lesion = False
      surf_plotter.subplot(i,0)
      flatmap.generic_surface_plot('lh',ax=surf_plotter)
      flatmap.surfaceplot_additional_ROI('SCAN','lh',ax=surf_plotter,roi_opacity=1.0)
      flatmap.surfaceplot_additional_ROI(target_key,'lh',ax=surf_plotter,roi_opacity=0.5,outline=outline_lesion)
      surf_plotter.add_text(f'{sub} L', font_size=12)
      
      surf_plotter.subplot(i,1)
      flatmap.generic_surface_plot('rh',ax=surf_plotter)
      flatmap.surfaceplot_additional_ROI('SCAN','rh',ax=surf_plotter,roi_opacity=1.0)
      flatmap.surfaceplot_additional_ROI(target_key,'rh',ax=surf_plotter,roi_opacity=0.5,outline=outline_lesion)
      surf_plotter.add_text(f'{sub} R', font_size=12)

surf_plotter.subplot(len(subjects),0)
fsav.set_plotter_surfaces(conte_surfs)
fsav.generic_surface_plot('lh',ax=surf_plotter)
fsav.surfaceplot_additional_ROI(SCAN_KEY,'lh',ax=surf_plotter,roi_opacity=0.5)
fsav.surfaceplot_additional_ROI(SCAN_KEY,'lh',ax=surf_plotter,outline=True)
for r in rois:
      fsav.surfaceplot_additional_ROI(r,'lh',ax=surf_plotter,roi_opacity=0.75)
      fsav.surfaceplot_additional_ROI(r,'lh',ax=surf_plotter,outline=True)
surf_plotter.add_text('fsav_fsLR32k L', font_size=12)

surf_plotter.subplot(len(subjects),1)
fsav.generic_surface_plot('rh',ax=surf_plotter)
fsav.surfaceplot_additional_ROI(SCAN_KEY,'rh',ax=surf_plotter,roi_opacity=0.5)
fsav.surfaceplot_additional_ROI(SCAN_KEY,'rh',ax=surf_plotter,outline=True)
for r in rois:
      fsav.surfaceplot_additional_ROI(r,'rh',ax=surf_plotter,roi_opacity=0.75)
      fsav.surfaceplot_additional_ROI(r,'rh',ax=surf_plotter,outline=True)
surf_plotter.add_text('fsav_fsLR32k R', font_size=12)
surf_plotter.link_views([0,2,4])
surf_plotter.link_views([1,3,5])

plot_config = [[("lh", i, 0, "Left"), ("rh", i, 1, "Right")] for i in range(3)]
plot_config = list(np.reshape(plot_config,(-1,4)))
surf_plotter.plot_config = plot_config
img_scaling = 10
orient = surf_plotter.swap_2_lateral()
surf_plotter.screenshot(str(figure_root / f'RFA_projection_surfaces_{orient}.png'),scale=5)
surf_plotter.save_graphic(figure_root / f'RFA_projection_surfaces_{orient}.svg')

orient = surf_plotter.swap_2_medial()
surf_plotter.screenshot(str(figure_root / f'RFA_projection_surfaces_{orient}.png'),scale=5)
surf_plotter.save_graphic(figure_root / f'RFA_projection_surfaces_{orient}.svg')

orient = surf_plotter.swap_2_iso()
surf_plotter.screenshot(str(figure_root / f'RFA_projection_surfaces_{orient}.png'),scale=5)
surf_plotter.save_graphic(figure_root / f'RFA_projection_surfaces_{orient}.svg')

for job in spin_jobs:
      subject = str(job['subject'])
      flatmap = job['flatmap']
      merged_scan_data = job['merged_scan_data']
      scan_map_labels = job['scan_map_labels']
      lesion_keys = job['lesion_keys']
      for lesion_key in lesion_keys:
            lesion_data = flatmap.additional_ROIs[lesion_key]
            multiclass_spin_model = SpinNullModel(flatmap.atlas, lesion_data, merged_scan_data)
            multiclass_spin_model.map_labels = scan_map_labels
            multiclass_spin_cache_path = spin_cache_root / subject / lesion_key / 'multiclass_scan_subrois'
            multiclass_spin_result = multiclass_spin_model.run_spintest_individual(
                  n_iterations=num_spin_iters,
                  use_stored_spins=True,
                  regenerate_spins=regenerate_spins,
                  spin_cache_path=multiclass_spin_cache_path
            )
            print(f'\n{subject} | multiclass lesion {lesion_key} vs merged SCAN subROIs')
            spin_plot_path = spin_cache_root / subject / lesion_key / f'{lesion_key}_vs_SCAN_subrois'
            multiclass_spin_result.correct_multiple_comparisons()
            multiclass_spin_result.plot(spin_plot_path, title=f'{subject}_{lesion_key}')

plt.show(block=False)
surf_plotter.show()
