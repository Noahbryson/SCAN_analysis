import os
from pathlib import Path
from src.SCAN_SingleSessionAnalysis import *
import matplotlib.pyplot as plt
import sys
from PyBrain.modules.VERA_PyBrain import PyBrain
from src.functions.graphics import default_gradient


import platform
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
    dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
    dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"


subject = 'BJH079_postRF'
subject = 'SLCH034'
# subject = 'BJH041'
gammaRange = [70,170]
brainType = "MNIbrain_destrieux"
# brainType = "aparc.a2009s"
# brainType = "patient_brain"
# brainType = "MNIbrain"
# session = 'pre_ablation'
# session = 'post_ablation'
session = 'aggregate'
aggpath = dataPath / 'Aggregate' / f'{subject}_{session}'

laplacian = False
bipolar = True
loadData=True
save = False
side = 'both'

if bipolar:
    reref = 'bipolar'
elif laplacian:
    reref = 'laplacian'
else:
    reref = 'common'

bp = dataPath/subject/'brain'/f'{brainType}.mat'
# brain = PyBrain(bp,subject=subject,brainName=brainType)
# brain._makeBipolarElectrodes()
# brain = PyBrain(bp,subject=subject,brainName=brainType,bipolar=bipolar,laplacian=laplacian)
try: #Generation of brain if the file exists, if not data will be parsed as is
    brain = PyBrain(bp,subject=subject,brainName=brainType,bipolar=bipolar,laplacian=laplacian,loadMapping=True)
    print(f'\nLoaded {brainType} for {subject} from {bp}\n')
    
    electrodemap,regionsLocs = brain._get_ROI_map()
    centralSulcusInfo = [[i,brain.regionColors[j]] for j,i in enumerate(brain.regions) if i.lower().find('central')>-1]
    plottingRegions = []
    for idx,i in enumerate(brain.regions):
        if i.lower().find('central')>-1:
                # plottingRegions.append([i,brain.regionColors[idx]])
                plottingRegions.append([i,[.7,.09,.09]])
        else:
                plottingRegions.append([i,[.9,.9,.9]])
                # plottingRegions.append([i,[.6,.6,.6]])
    side = 'both'
    brainFlag = True
    
    ROIs = [i for i in brain.regions if i.lower().find('central')>-1]
    ROIs.extend(['G_front_inf-Opercular','G_insular_short'])
    brain.get_ROI_electrodes(ROIs,savename='Motor-Strip')

    ROIs = [i for i in brain.regions if i.lower().find('precentral')>-1]
    ROIs.extend(['S_central'])
    brain.get_ROI_electrodes(ROIs,savename='precentral&central')
    
    
except Exception as err:
    brainFlag = False
    print(err)
    brain = None
    print('brain path not valid')

print(f'Running Analysis on Patient: {subject}\nsession: {session}')
if subject == 'BJH041':
    remove = ['OR']
else: remove=[]
a = SCAN_SingleSessionAnalysis(dataPath,subject,session,remove_trajectories=remove,
    load=loadData,plot_stimuli=True,gammaRange=gammaRange,refType=reref)

a.plot_session_EMG()
# plt.show()
a.save_movement_latencies()
a.plot_movement_latencies()
plt.show(block=False)
# a.run_ERP_processing(plot=True,save=True,show=False)
r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(save=save,makePlots=save)
sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals,alpha=0.05)
effect_of_interest =r_sq
effect_name = 'r_sq'
datasubset = sig_chans
intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(effect_of_interest,[],
                        save=save,label='significant',thresh=.1,comparison='')
channel_classifcation_out = []
if brainFlag:
    electrodemap,regionsLocs = brain._get_ROI_map()
    for i in channel_classifcation:
        if i in sig_chans:
            channel_classifcation_out.append(f'{i},{channel_classifcation[i]},1,{electrodemap[i]}\n')
        else: 
            channel_classifcation_out.append(f'{i},{channel_classifcation[i]},0,{electrodemap[i]}\n')
with open(aggpath/'channel_classifications.csv','w') as fp:
    fp.write('channel,class,significant,region\n')
    fp.writelines(channel_classifcation_out)
intereffectors = [i.replace('_','') for i in intereffectors]
nonspecifics = [i.replace('_','') for i in nonspecifics]

cmap_resolution=1

tuning_colors,target_colors = default_gradient(cmap_resolution,run_circular=False)
tuning,chan_tuning_colors,angle_key, color_array = a.somatotopic_tuning(r_sq,tuning_colors=tuning_colors,plotCMAP=True)
colorLab = list(angle_key.keys())
labColor = [i['color'] for i in angle_key.values()]
t_ = [[i,j['color']] for i,j in angle_key.items()]
t_names = [i[0] for i in t_]
t_colors = [i[1] for i in t_]
allChans = tuning['channel'].to_list()
shared_rep = a.shared_representation(effect_of_interest,sig_chans)
a.run_ERP_processing(show=False,save=True,beta=True)

showFlag = True
if brainFlag and showFlag:
    allChans_idx = brain._getElectrodeIndexFromLabels(labels=allChans)
    v1=brain.generateAxis(1)
    v1,_=brain.plotBrainVolume(v1,0.05,color=[1,1,1],side=side)
    v1=brain._plotBrainRegions(v1,regions=[i[0] for i in centralSulcusInfo], colors=[i[-1] for i in centralSulcusInfo],opacity=.2,side=side)
    
    nonspecifics_locs = [electrodemap[i] for i in nonspecifics]
    intereffector_locs = [electrodemap[i] for i in intereffectors]
    print(channel_classifcation)
    print('intereffectors')
    for i,j in zip(intereffectors,intereffector_locs):
            print(i,': ',j)
    print('\n\nnonspecifics')
    # non_specific= ['inter','foot-face','hand-face','hand-foot']
    # multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
    # targets = []
    
    # for i,j in zip(multiMotor,nonspecifics_locs):
    #         # if j.find('precentral')>-1 or j.find('S-central')>-1:
    #         if j.find('central')>-1:
    #             print(i[0],': ',j,' ',i[1])
    #             targets.append(i[0])
    non_specific= ['inter','foot-face','hand-face','hand-foot']
    multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
    targets = []
    for i in (multiMotor):
        print(i[0],': ',i[1])
        targets.append(i[0])
    
    targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
    targetIdx.extend( brain._getElectrodeIndexFromLabels(labels=nonspecifics))
    targetIdx = brain._getElectrodeIndexFromLabels(labels=targets)
    v1,aa=brain.plot_electrodes_on_volume(v1,electrodeSubset=targetIdx,side=side,color=(0,1,1))
    targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
    v1,ab=brain.plot_electrodes_on_volume(v1,electrodeSubset=targetIdx,side=side,color=(1,1,0))
    v1.show()
    targets = intereffectors
    targetIdx = brain._getElectrodeIndexFromLabels(labels=targets)
    
    rma_vol=brain.generateAxis()
    rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in plottingRegions], 
            colors=[i[-1] for i in plottingRegions],opacity=0.7,side=side)
    rma_vol=brain._plotBrainRegions(rma_vol,regions=[i[0] for i in centralSulcusInfo], 
            colors=[i[-1] for i in centralSulcusInfo],opacity=1,side=side)
    targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
    rma_vol,ab=brain.plot_electrodes_on_volume(rma_vol,electrodeSubset=targetIdx,side=side,color=(1,1,0))

    # rma_vol=brain._plotEffectOnVolume()
    
    rma_vol.show_axes()
    
    rma_vol.show()
    # v1.show()
else:
    print('intereffectors')
    for i in intereffectors:
            print(i)
    print('\n\nnonspecifics')
    non_specific= ['inter','foot-face','hand-face','hand-foot']
    multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
    targets = []
    for i in (multiMotor):
        print(i[0],': ',i[1])
        targets.append(i[0])
    
plt.show()