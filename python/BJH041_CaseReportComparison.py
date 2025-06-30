from pathlib import Path
import os
import numpy as np
from src.SCAN_group_analysis import SCAN_group_analysis, compute_effect_centroid, ablation_effect
from matplotlib import pyplot as plt
import distinctipy
import random
import platform
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain
from VERA_PyBrain.modules.MRI_Utils import load_fsLabel, combine_volumes
lesion_volumes = [Path('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/patients/BJH041/MRI/post_RF/RF_Volume_v2.label'),Path('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/patients/BJH041/MRI/post_RF/RF_Edema_v2.label')]

lesion_volumes = [load_fsLabel(i) for i in lesion_volumes]
lesion = combine_volumes(lesion_volumes)
print(f' combined lesion {lesion[0]}')
subjectSessions = ['BJH041_pre_ablation','BJH041_post_ablation']
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
      dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"

subject = 'BJH041'
brainType = "MNIbrain_destrieux"
brainType = "brain_cortex"
laplacian = False
bipolar = True

if bipolar:
      reref = 'bipolar'
elif laplacian:
      reref = 'laplacian'
else:
      reref = 'common'
bp = dataPath/subject/'brain'/f'{brainType}.mat'
try:
      brain = PyBrain(bp,subject=subject,brainName=brainType,loadMapping=True)
      print(f'\nLoaded {brainType} for {subject} from {bp}\n')
      
except:
      brain = None
      print('brain path not valid')
electrodemap,regionsLocs = brain._get_ROI_map()
centralSulcusInfo = [[i,brain.regionColors[j]] for j,i in enumerate(brain.regions) if i.lower().find('central')>-1]
side = 'L'
brain.add_volume(lesion_volumes,['rfa','edema'])
brain.add_volume(lesion,'lesion',triangulate=True)
indices = [[1, 2, 0],[1, 0, 2]]
ax = brain._generateAxis()
ax.show_axes_all()

brain._plotBrainVolume(ax,opacity=0.01,color=[0.2,0.2,0.2])
brain._plotAdditionalVolume(ax,'lesion',opacity=0.2,color=(255,0,0))
print(brain.volumes['lesion'])
brain.plot_all_traj(ax=ax)

ax.close()


a = SCAN_group_analysis(dataPath/'Aggregate',subjectSessions)
intereffectors = ['HL3-b-4','HL4-b-5','HL8-b-9','HL9-b-10','IL1-b-2','KL10-b-11','KL11-b-12','KL12-b-13','LL5-b-6']
ablated = ['KL11-b-12','KL12-b-13','KL13-b-14','KL14-b-15','KL15-b-16']
ROIs = [i for i in brain.regions if i.lower().find('central')>-1]
ROIs.extend(['G_front_inf-Opercular','G_insular_short'])
# ROIs = brain.regions
ROI_info = [[i,brain.regionColors[j]] for j,i in enumerate(brain.regions) if i in ROIs]
# a.compare_EMG_isolation(subject_sessions[0],subject_sessions[1])
# a.compare_tuning(subject_sessions[0],subject_sessions[1],region_search_strs=['central','G_front_inf-Opercular','G_insular_short'],paired=True,effect='Magnitude',)
# a.compare_tuning(subject_sessions[0],subject_sessions[1],region_search_strs=['central','G_front_inf-Opercular','G_insular_short'],paired=True,effect='Angle')
# a.compare_rsq(subject_sessions[0],subject_sessions[1],region_search_strs='central',paired=True)



dat = a.get_subject_session(subjectSessions[1])
centeroids, ROI_effects = compute_effect_centroid(brain=brain,effect=dat,ROIs=ROIs,plot=True)
dat = a.get_subject_session(subjectSessions[0])
data= a.compare_shared_rep(subjectSessions[0],subjectSessions[1],region_search_strs=ROIs,paired=True,channel_markers = ablated,significant=False)
centeroids, ROI_effects = compute_effect_centroid(brain=brain,effect=dat,ROIs=ROIs,plot=True)
plt.show()
ablation_effect(brain,data,'Hand',brain.volumes['lesion'].points,'lesion',ROIs, centroid=False)
ablation_effect(brain,data,'Foot',brain.volumes['lesion'].points,'lesion',ROIs, centroid=False)
ablation_effect(brain,data,'Tongue',brain.volumes['lesion'].points,'lesion',ROIs, centroid=False)
ablation_effect(brain,data,'shared_rep',brain.volumes['lesion'].points,'lesion',ROIs, centroid=False)
plt.show()
lesion = lesion[:,[1, 0, 2]]
brain.add_foci(np.array([i for i in centeroids.values()]),[i for i in centeroids])
ax = brain._generateMultiAxis(1,3,link=True)
ax.subplot(0,0)
brain._plotBrainVolume(ax,lighting=True,color=[0.1,0.1,0.1],opacity=0.05)
brain.plot_all_traj()
brain._plot_foci_on_volume(ax)
brain._plotAdditionalVolume(ax,'lesion')
brain._plotEffectOnVolume(ax,ROI_effects.rename({'Hand':'metric'},axis=1))
ax.subplot(0,1)
brain._plotBrainVolume(ax,lighting=True,color=[0.1,0.1,0.1],opacity=0.05)
brain.plot_all_traj()
brain._plot_foci_on_volume(ax)
brain._plotAdditionalVolume(ax,'lesion')
brain._plotEffectOnVolume(ax,ROI_effects.rename({'Foot':'metric'},axis=1))
ax.subplot(0,2)
brain._plotBrainVolume(ax,lighting=True,color=[0.1,0.1,0.1],opacity=0.05)
brain.plot_all_traj()
brain._plot_foci_on_volume(ax)
brain._plotAdditionalVolume(ax,'lesion')
brain._plotEffectOnVolume(ax,ROI_effects.rename({'Tongue':'metric'},axis=1))
# plt.show(block=False)
plt.close()
ax.show()