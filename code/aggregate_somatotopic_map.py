import argparse
from PyBrain.modules.surface_projection import projectAtlas, Atlas, groupAtlas, adjust_lightness
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatch
from src.SCAN_group_analysis import SCAN_group_analysis
import pandas as pd
import os
from pathlib import Path
from platform import system
import distinctipy as dp
from src.functions.graphics import default_gradient
from PyBrain.modules.spin_nulls import SpinNullModel
from PyBrain.modules.statistics import NMI,dice_coefficient,multi_class_dice


def CLI_args() -> argparse.Namespace:
      """Parse command line arguments for the aggregate somatotopic map script."""
      parser = argparse.ArgumentParser()
      parser.add_argument(
            "--run-gui",
            action="store_true",
            default=False,
            help="Launch the ROI multiview GUI before running the spin tests.",
      )
      parser.add_argument(
            "--motor-rois",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="restrict ROIs to Motor Strip",
      )
      
      parser.add_argument(
            "--operculum",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="include operculum",
      )
      parser.add_argument(
            "--insula",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="include insula",
      )
      parser.add_argument(
            "--pool",
            action="store_true",
            default=False,
            help="pool all contacts to the left hemisphere",
      )

      parser.add_argument(
            "--regen-spins",
            action="store_true",
            default=False,
            help="regenerate spins rather than attempting to load from a saved file",
      )
      parser.add_argument(
            "--n-spins",
            type=int,
            default=1000,
            help="number of spins to generate and evaluate",
      )
      parser.add_argument(
            "--radius",
            type=float,
            default=6,
            help="electrode neighborhood radius in mm",
      )
      return parser.parse_args()


args = CLI_args()
print(args)
userpath = Path(os.path.expanduser('~'))
if system() == 'Windows':
      boxpath = userpath
else:
      boxpath = userpath / "Library/CloudStorage/Box-Box"

"""User Arguments"""

dataroot = boxpath/ 'Brunner Lab'/'DATA'/'SCAN_Mayo'
subjects_file = dataroot/'subjects.json'
atlas_res = '32k'
atlas: Atlas = Atlas.fs_LR_from_fsav(atlas_res)
atlas.pt_sphere_name = 'sphere.reg.surf.gii' 
subject = 'fsaverage_wb'
seg = userpath / 'Documents'/'NCAN'/'patients'/subject/'segmentation'

neighborhood_radius = args.radius  # 5mm electrode recording volume + 1mm to account for the radius of the electrode
print(f'Analysing maps at {neighborhood_radius}mm neighborhood radius')
num_spin_iters = args.n_spins
regenerate_spins = args.regen_spins
pool_hemispheres = args.pool
target_hemi = 'lh'


"""Analysis Script"""
spin_cache_path = Path(seg.parent /'gifti'/'spintest'/'motor_maps'/f'{neighborhood_radius}mm_radius')
a = SCAN_group_analysis(dataroot,subjects_file)
flatbrain = groupAtlas(subjects=a.subjects,template_dir = seg, atlas=atlas)
SCANMAP = Path('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/imaging/HCP_Spots_Effectors_CS.dtseries.nii')
SCAN_KEY = 'HCP_SCAN'
flatbrain.load_cifti_data(SCANMAP,'HCP_SCAN')
SCAN_VMAP = {0:'na', 1.5:'inter',10:'hand',11:'face',17:'foot'}
SCAN_CMAP = {'na':(0,0,0,0),'inter':(158/255,38/255,108/255,1),'hand':(68/255,1,1,1), 'face':(1,142/255,52/255,1),'foot':(32/255,133/255,44/255,1)}
flatbrain.update_additional_ROI_value_map(SCAN_KEY,SCAN_VMAP,SCAN_CMAP)
if pool_hemispheres: 
      flatbrain.project_electrodes_to_single_hemi(target_hemi=target_hemi,overrite_data=True)
metric = 'r-sq'
# metric = 'd'
power = a.load_task_power(metric_name=metric)
result = power.join(flatbrain.electrode_library)
nan_locs = result.index[result['dist'].isna()]
result = result.drop(index=nan_locs,errors='ignore')
colors = {i:j for i,j in zip(result['class'].unique(),dp.get_colors(len(result['class'].unique())))}


grad = np.array([[1,0,0], [0,1,0], [0,0,1]])
colors = {i:adjust_lightness(SCAN_CMAP[i][0:-1],0.6) for i in ['hand','foot','face']}
mixed_groups = ['hand-foot','hand-face','foot-face']
for i in mixed_groups:
      keys = i.split('-')
      c = [colors[j] for j in keys]
      mean_col = np.mean(c,axis=0)
      colors[i] = mean_col

colors['na'] = (0,0,0)
colors['inter'] = (1,0,1)

if args.motor_rois:
      print('restricting motor ROIs')
      ROIs = list(np.unique([i for i in flatbrain.electrode_library['region'] if i.lower().find('central')>-1]))
      
      insula_ROIs = list(np.unique(i for i in flatbrain.electrode_library['region'] if i.lower().find('insula')>-1))
      insula_ROIs.append('G_insular_short')
      if args.insula:
            print('including insula')
            ROIs.extend(insula_ROIs)
      if args.operculum:
            print('including operculum')
            ROIs.append('G_front_inf-Opercular')
else:
      ROIs = flatbrain.electrode_library['region']
ROIs = set(ROIs)
update_dict = {i:'inter' for i in mixed_groups}

result['color'] = result['class'].map(colors)
result['regionLoc'] = np.where(result['region'].isin(ROIs),1,0)
result = result.loc[result['regionLoc']==1]
result['alpha'] = result['class']
result['alpha'] = np.where(result['class'] == 'na', 0.05, 0.9)

agg_data = result
agg_data['class'] = agg_data['class'].replace(update_dict) 

scan_neighbor_maps, scan_label_classes, label_id_map = flatbrain.euclidean_neighborhood_voting_map(agg_data,r=neighborhood_radius,sigma=2.5)
cmap = flatbrain.make_cmap_from_vmap(label_id_map)
cmap.update(SCAN_CMAP)
ephys_map = "motor_maps"
flatbrain.add_additional_ROI_value_map(ephys_map,scan_label_classes,label_id_map,color_map=cmap)


if args.run_gui:
      flatbrain.launch_roi_multiview_gui(ephys_map,block=True)

"""Arguments for spintest"""
fp = boxpath / 'Brunner Lab/DATA/SCAN_Mayo/Aggregate/GroupMaps'

key_subset = list(SCAN_CMAP.keys())
key_subset.pop(key_subset.index('na'))
emap = flatbrain.return_ROI_mapping(ephys_map)
imap = flatbrain.return_ROI_mapping(SCAN_KEY)

spin_model = SpinNullModel(flatbrain.atlas,emap,imap)

# spin_model.map_labels = key_subset
spin_model.map_labels = {i:key_subset for i in spin_model.hemis}
if pool_hemispheres: spin_cache_path = spin_cache_path.parent / f'{spin_cache_path.name}_pooled'
spin_model.generate_spins(n_spins=num_spin_iters,decim=10,spin_cache_path=spin_cache_path,overwrite=regenerate_spins)

"""spin test execution"""
spinname = 'ephys_vs_SCAN_motor-subset'
if pool_hemispheres: spinname = 'pooled_' + spinname
spintest = spin_model.run_spintest(method=multi_class_dice,n_iterations=num_spin_iters,input_masking=key_subset)
spintest.correct_multiple_comparisons()
spintest.plot(title=spinname)
# spintest.toJson(fp,decorator=spinname)

spinname = 'ephys_vs_SCAN_motor-subset_individual-classes'
if pool_hemispheres: spinname = 'pooled_' + spinname
spintest = spin_model.run_spintest_individual(n_iterations=num_spin_iters,input_masking=key_subset)
spintest.correct_multiple_comparisons()
spintest.plot(title=spinname)
# spintest.toJson(fp,decorator=spinname)

"""end spintest"""

# surf = flatbrain.surfaceplot_additional_ROI(ephys_map,'lh',showLegend=True)
"""Electrode Specific Plot"""
# result = agg_data
temp = result
result['class'] = result['class'].replace(update_dict) 
result['color'] = result['class'].map(colors)
result['alpha'] = result['class']
result['alpha'] = np.where(result['class'] == 'na', 0.05, 0.9)
eLeft = result.loc[result['hemi']=='lh']
# eLeft['color'] = eLeft['class'].map(update_dict)
colors_l = eLeft['color'].to_list()
alpha_l = eLeft['alpha'].to_list()
subset_l = ['names',eLeft.index]
aL = flatbrain.flatmap_plot('L',annot=False,outline=False)
aL = flatbrain.flatplot_additional_ROI(SCAN_KEY,'L',aL)
flatbrain.plot_electrodes(ax=aL,hemi='lh',bipolar=True,subset=subset_l, color=colors_l,alphas=alpha_l)
patch_leg = []
for i,j in colors.items():
      patch_leg.append(mpatch.Patch(color=j, label=i))
aL.legend(handles=patch_leg)

aR = flatbrain.flatmap_plot('R',annot=False,outline=False,)
eRight = result.loc[result['hemi'] == 'rh']
alpha_r = eRight['alpha'].to_list()
colors_r = eRight['color'].to_list()
subset_r = ['names',eRight.index]
aR = flatbrain.flatplot_additional_ROI(SCAN_KEY,'R',aR)
flatbrain.plot_electrodes(ax=aR,hemi='rh',bipolar=True,subset=subset_r, color=colors_r,alphas=alpha_r)
aR.legend(handles=patch_leg)

flatbrain.fit_image_to_electrode(aR,eRight)
flatbrain.fit_image_to_electrode(aL,eLeft)


"""ROI maps only"""
plt.figure()
aL = flatbrain.flatplot_additional_ROI(ephys_map,'L',showLegend=True)
flatbrain.flatplot_additional_ROI(SCAN_KEY,'L',ax=aL,outline=True)
flatbrain.fit_image_to_electrode(aL,eLeft)

plt.figure()
aR = flatbrain.flatplot_additional_ROI(ephys_map,'R',showLegend=True)
flatbrain.flatplot_additional_ROI(SCAN_KEY,'R',ax=aR,outline=True)
flatbrain.fit_image_to_electrode(aR,eRight)



figsavepath = dataroot / 'group_figs'
# os.makedirs(figsavepath,exist_ok=True)
roi_tag = f"motor-rois_{int(args.motor_rois)}_insula_{int(args.insula)}_operculum_{int(args.operculum)}"
# figL.savefig(figsavepath / f'group_map_L_electrodes_{roi_tag}.png',transparent=1)
# figR.savefig(figsavepath / f'group_map_R_electrodes_{roi_tag}.png',transparent=1)





"""Save all open figs as both png and svg"""

figsavepath = fp / 'figures' / f'{num_spin_iters}-iter_{neighborhood_radius}mm_rad_{roi_tag}'
figsavepath.mkdir(parents=True, exist_ok=True)
for fig_num in plt.get_fignums():
      fig = plt.figure(fig_num)
      outfile = figsavepath / f'aggregate_somatotopic_map_fig{fig_num}.png'
      fig.savefig(outfile, dpi=300, bbox_inches='tight',transparent=False)
      outfile = figsavepath / f'aggregate_somatotopic_map_fig{fig_num}.svg'
      fig.savefig(outfile)
print(f'saved to {figsavepath}')
plt.close('all')




# surf = flatbrain.generic_surface_plot('lh')

plot_class = 'inter'

