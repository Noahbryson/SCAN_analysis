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
# brainType = "brain_cortex"
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
brain.add_volume(lesion_volumes[0],'rfa')
brain.add_volume(lesion_volumes[1],'edema')
brain.add_volume(lesion,'lesion',triangulate=True)
indices = [[1, 2, 0],[1, 0, 2]]
# ax = brain._generateAxis()
# ax.show_axes_all()

# brain._plotBrainVolume(ax,opacity=0.01,color=[0.2,0.2,0.2])
# brain._plotAdditionalVolume(ax,'lesion',opacity=0.2,color=(255,0,0))
# print(brain.volumes['lesion'])
# brain.plot_all_traj(ax=ax)

# ax.close()


a = SCAN_group_analysis(dataPath/'Aggregate',subjectSessions)
from src.SCAN_group_analysis import plot_latencies
method = 'cluster'
ERP_compare = a.compare_ERP_pair(subjectSessions[0],subjectSessions[1],method=method,savePath=f'/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/Aggregate/pairs/BJH041_pre-v-post/{method}')

breakpoint()
fig = plot_latencies(a.load_latencies())
suppPath = r'/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/zSUPP/reaction_time'
os.makedirs(suppPath,exist_ok=True)
fig.savefig(f'{suppPath}/latencies.svg')
plt.show()
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


plt.rcParams["figure.figsize"] = (18,10)
suppPath = r'/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/zSUPP/effect_falloff'
data= a.compare_shared_rep(subjectSessions[0],subjectSessions[1],region_search_strs=ROIs,paired=True,channel_markers = ablated,significant=False)

dat = a.get_subject_session(subjectSessions[1])
centeroids, ROI_effects, fig = compute_effect_centroid(brain=brain,effect=dat,ROIs=ROIs,plot=True)
# fig.savefig(fname=f'{suppPath}/post_ablation_centroids.svg',transparent=True,format='svg')
dat = a.get_subject_session(subjectSessions[0])
centeroids, ROI_effects, fig = compute_effect_centroid(brain=brain,effect=dat,ROIs=ROIs,plot=True)
# fig.savefig(fname=f'{suppPath}/pre_ablation_centroids.svg',transparent=True,format='svg')

lesionFig, lesionAx = plt.subplots(nrows=1,ncols=4,sharex=True,sharey=True,num='Lesion Effect Fallof')
plot_effects = ['Hand','Foot','Tongue','shared_rep']
for i,a in zip(plot_effects,lesionAx):
      ablation_effect(brain,data,i,brain.volumes['lesion'].points,'lesion',ROIs, centroid=False,ax=a)
plt.show()
lesionFig.savefig(fname=f'{suppPath}/ablationFalloff.svg',transparent=True,format='svg')
# lesionFig.savefig(fname=f'{suppPath}/ablationFalloff_log.svg',transparent=True,format='svg')
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
lesion = lesion[:,[1, 0, 2]]
brain.add_foci(np.array([i for i in centeroids.values()]),[i for i in centeroids])
ax = brain._generateMultiAxis(1,3,link=True)
ax.subplot(0,0)
brain.plotBrainVolume(ax,lighting=True,color=[0.1,0.1,0.1],opacity=0.05)
brain.plot_all_traj()
brain._plot_foci_on_volume(ax)
brain.plotAdditionalVolume(ax,'lesion')
brain._plotEffectOnVolume(ax,ROI_effects.rename({'Hand':'metric'},axis=1))
ax.subplot(0,1)
brain.plotBrainVolume(ax,lighting=True,color=[0.1,0.1,0.1],opacity=0.05)
brain.plot_all_traj()
brain._plot_foci_on_volume(ax)
brain.plotAdditionalVolume(ax,'lesion')
brain._plotEffectOnVolume(ax,ROI_effects.rename({'Foot':'metric'},axis=1))
ax.subplot(0,2)
brain.plotBrainVolume(ax,lighting=True,color=[0.1,0.1,0.1],opacity=0.05)
brain.plot_all_traj()
brain._plot_foci_on_volume(ax)
brain.plotAdditionalVolume(ax,'lesion')
brain._plotEffectOnVolume(ax,ROI_effects.rename({'Tongue':'metric'},axis=1))
plt.show(block=False)

# ax.show()
ax.close()