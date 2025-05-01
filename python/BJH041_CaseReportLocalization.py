import os
from pathlib import Path
from src.SCAN_SingleSessionAnalysis import *
import matplotlib.pyplot as plt
import sys
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain
from src.functions.graphics import circular_gradient3,hex2RGB,targetColorSwatch,default_gradient, circle_gradient_key
import platform




localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      boxPath = userPath/"Box"
else:
      boxPath = userPath/"Library/CloudStorage/Box-Box"
dataPath = boxPath / "Brunner Lab/DATA/SCAN_Mayo"
subject = 'BJH041'
gammaRange = [70,170]
session = 'pre_ablation'
brainType = "MNIbrain_destrieux"

laplacian = False
bipolar = True
loadData=True
save = False

if bipolar:
      reref = 'bipolar'
elif laplacian:
      reref = 'laplacian'
else:
      reref = 'common'

bp = dataPath/subject/'brain'/f'{brainType}.mat'
try:
      brain = PyBrain(bp,subject=subject,brainName=brainType)
      print(f'\nLoaded {brainType} for {subject} from {bp}\n')
      
except:
      brain = None
      print('brain path not valid')

electrodemap,regionsLocs = brain._get_ROI_map()
centralSulcusInfo = [[i,brain.regionColors[j]] for j,i in enumerate(brain.regions) if i.lower().find('central')>-1]

side = 'both'

# traj_vol = brain.plot_all_traj()
# traj_path = boxPath/r'Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/FIG1/traj'
# brain.glassbrain_figure_export(traj_vol,savepath=traj_path,figname='trajectories')
# traj_vol.close()
side = 'L'

# brain.projection_2D('xy',[],'coronal')

a = SCAN_SingleSessionAnalysis(dataPath,subject,session,remove_trajectories=['OR'],
      load=loadData,plot_stimuli=False,gammaRange=gammaRange,refType=reref)
r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(saveMAT=save)
sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals,alpha=0.05)
cmap_resolution=1
tuning_colors = default_gradient(cmap_resolution)
tuning,chan_tuning_colors,angle_key, color_array = a.somatotopic_tuning(r_sq,tuning_colors=tuning_colors,plotCMAP=False,cmapResolution=cmap_resolution)
# for i,j in tuning_colors.items():
      # print(i,j)
colorLab = list(angle_key.keys())
labColor = [i['color'] for i in angle_key.values()]

tuning_path = boxPath/r'Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/FIG2/tuning'
t_ = [[i,j['color']] for i,j in angle_key.items()]
t_names = [i[0] for i in t_]
t_colors = [i[1] for i in t_]



effect_of_interest = r_sq
effect_name = 'r_sq'
datasubset = sig_chans
allChans = tuning['channel'].to_list()
allChans_idx = brain._getElectrodeIndexFromLabels(labels=allChans)
shared_rep = a.shared_representation(effect_of_interest,sig_chans)

# intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(d_res,datasubset,save=save,label='significant',thresh=1,comparison='')
# intereffectors = [i.replace('_','') for i in intereffectors]
# intereffector_locs = [electrodemap[i] for i in intereffectors]
# print("intereffectors (cohen's d)")
# for i,j in zip(intereffectors,intereffector_locs):
#       print(i,': ',j)      

intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(effect_of_interest,datasubset,save=save,label='significant',thresh=.1,comparison='')
intereffectors = [i.replace('_','') for i in intereffectors]
nonspecifics = [i.replace('_','') for i in nonspecifics]
nonspecifics_locs = [electrodemap[i] for i in nonspecifics]
intereffector_locs = [electrodemap[i] for i in intereffectors]
# print(channel_classifcation)
print('intereffectors')
for i,j in zip(intereffectors,intereffector_locs):
      print(i,': ',j)
print('\n\nnonspecifics')
non_specific= ['inter','foot-face','hand-face','hand-foot']
multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
targets = []
for i,j in zip(multiMotor,nonspecifics_locs):
      if j.find('central')>-1:
            print(i[0],': ',j,' ',i[1])
            targets.append(i[0])
# targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
# targetIdx.extend( brain._getElectrodeIndexFromLabels(labels=nonspecifics))

plottingRegions = []
for idx,i in enumerate(brain.regions):
      if i.lower().find('central')>-1:
            # plottingRegions.append([i,brain.regionColors[idx]])
            # plottingRegions.append([i,[.7,.09,.09]])
            # plottingRegions.append([i,[1,.7,.09]])
            plottingRegions.append([i,[.71,.31,.01]])
      else:
            plottingRegions.append([i,[.9,.9,.9]])
            # plottingRegions.append([i,[.6,.6,.6]])

fig1=True
if fig1:
      SZ_channels = ['HL6-b-7','HL7-b-8','HL8-b-9','HL9-b-10','HL10-b-11','HL11-b-12','KL8-b-9','KL9-b-10','KL10-b-11','KL11-b-12','KL12-b-13','KL13-b-14','KL14-b-15','KL15-b-16','ML2-b-3','ML3-b-4','ML4-b-5','ML5-b-6','ML6-b-7']
      sz_colors = np.zeros([len(SZ_channels),3])
      sz_colors[2:4,:]   = np.array([53, 152, 204])/255
      # sz_colors[11,:]    = np.array([0,1,0])
      sz_colors[11,:]    = np.array([1,.02,.1])
      sz_colors[12:14,:] = np.array([105,201,202])/255
      sz_colors[15:18,:] = np.array([75,96,247])/255
      SZ_colors = {i:j for i,j in zip(SZ_channels,sz_colors)}
      sz_vol=brain._generateAxis(1)
      sz_vol,coords=brain._plotBrainVolume(sz_vol,0.05,[1,1,1],side=side)
      SZ_path = boxPath/r'Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/FIG1'
      targetIdx = brain._getElectrodeIndexFromLabels(labels=SZ_channels)
      sz_vol=brain._generateAxis()
      sz_vol=brain._plotBrainRegions(sz_vol,regions=[i[0] for i in centralSulcusInfo], colors=[i[-1] for i in centralSulcusInfo],opacity=1,side=side)
      sz_vol=brain._plotBrainRegions(sz_vol,regions=[i[0] for i in plottingRegions], 
            colors=[i[-1] for i in plottingRegions],opacity=.7,side=side)
      sz_vol.show_axes()
      
      testData = shared_rep.rename({'Significant':'p'})
      testData['metric'] = np.ones(len(testData))*0.15
      sz_vol= brain.opaquebrain_projection_figure_export(sz_vol,SZ_path,'SZ_network_prop',data=testData,
            electrodeSubset=targetIdx,significant=False,colorMap=[SZ_colors[i] for i in SZ_channels])
      sz_vol.show()
      
fig2_brains = False
if fig2_brains:            
      plottingRegions = []
      for idx,i in enumerate(brain.regions):
            if i.lower().find('central')>-1:
                  # plottingRegions.append([i,brain.regionColors[idx]])
                  plottingRegions.append([i,[.7,.09,.09]])
            else:
                  plottingRegions.append([i,[.9,.9,.9]])
                  # plottingRegions.append([i,[.6,.6,.6]])

      """Shared Representation"""
      sharedRep_path = boxPath/r'Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/FIG2/shared_rep'
      sharedRep_vol=brain._generateAxis()
      sharedRep_vol=brain._plotBrainRegions(sharedRep_vol,regions=[i[0] for i in plottingRegions], 
                                    colors=[i[-1] for i in plottingRegions],opacity=0.7,side=side)
      sharedRep_vol=brain._plotBrainRegions(sharedRep_vol,regions=[i[0] for i in centralSulcusInfo], 
                        colors=[i[-1] for i in centralSulcusInfo],opacity=1,side=side)
      sharedRep_vol.show_axes()
      sharedRep_vol= brain.opaquebrain_projection_figure_export(sharedRep_vol,sharedRep_path,'shared_representation',data=shared_rep.rename({'Shared Rep':'metric','Significant':'p'},axis=1),
                        electrodeSubset=allChans_idx,significant=True)
      
      sharedRep_vol.close()
      """Somatotopic Tuning with Shared Representation as Magnitude"""
      fig = circle_gradient_key(tuning_colors,target_names=t_names,target_colors=t_colors)
      plt.show(block=False)
      tuning_vol=brain._generateAxis()
      tuning_vol=brain._plotBrainRegions(tuning_vol,regions=[i[0] for i in plottingRegions],
                                    colors=[i[-1] for i in plottingRegions],opacity=0.7,side=side)
      tuning_vol=brain._plotBrainRegions(tuning_vol,regions=[i[0] for i in centralSulcusInfo], 
                                    colors=[i[-1] for i in centralSulcusInfo],opacity=1,side=side)
      tuning_vol.show_axes()
      tuning_vol = brain.opaquebrain_projection_figure_export(tuning_vol,tuning_path,'somatotopic_tuning',data=shared_rep.rename({'Shared Rep':'metric','Significant':'p'},axis=1),
                  electrodeSubset=allChans_idx,significant=True, colorMap=[chan_tuning_colors[i] for i in allChans])
      tuning_vol.close()
      # tuning_vol.show()
      fig.savefig(tuning_path/'colorwheel.svg',format='svg')
      plt.close(fig)

      """K Traj RMA Tuning"""      
      rma_vol=brain._generateAxis(1)
      rma_vol,coords=brain._plotBrainVolume(rma_vol,0.05,[1,1,1],side=side)
      rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in centralSulcusInfo], colors=[i[-1] for i in centralSulcusInfo],opacity=.02,side=side)
      RMA_path = boxPath/r'Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/FIG2/RMA'
      
      targets = [i for i in intereffectors if i.find('KL')>=0]
      targetIdx = brain._getElectrodeIndexFromLabels(labels=targets)
      rma_vol=brain._generateAxis()
      rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in plottingRegions], 
            colors=[i[-1] for i in plottingRegions],opacity=0.7,side=side)
      rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in centralSulcusInfo], 
            colors=[i[-1] for i in centralSulcusInfo],opacity=1,side=side)
      rma_vol.show_axes()
      rma_vol= brain.opaquebrain_projection_figure_export(rma_vol,RMA_path,'target_RMA_tuning',data=shared_rep.rename({'Shared Rep':'metric','Significant':'p'},axis=1),
            electrodeSubset=targetIdx,significant=True,colorMap=[chan_tuning_colors[i] for i in targets])
      rma_vol.close()
      
      """All RMA Tuning"""
      targets = intereffectors
      targetIdx = brain._getElectrodeIndexFromLabels(labels=targets)
      
      rma_vol=brain._generateAxis()
      rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in plottingRegions], 
            colors=[i[-1] for i in plottingRegions],opacity=0.7,side=side)
      rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in centralSulcusInfo], 
            colors=[i[-1] for i in centralSulcusInfo],opacity=1,side=side)
      rma_vol.show_axes()
      rma_vol= brain.opaquebrain_projection_figure_export(rma_vol,RMA_path,'RMA_tuning',data=shared_rep.rename({'Shared Rep':'metric','Significant':'p'},axis=1),
            electrodeSubset=targetIdx,significant=True,colorMap=[chan_tuning_colors[i] for i in targets])
      rma_vol.close()
print(0)
