import os
from pathlib import Path
from src.SCAN_SingleSessionAnalysis import *
import matplotlib.pyplot as plt
import sys
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain

import platform
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
      dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"
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
if bipolar:
            brain._makeBipolarElectrodes()
if laplacian:
      pass
electrodemap,regionsLocs = brain._get_ROI_map()
centralSulcusInfo = [[i,brain.regionColors[j]] for j,i in enumerate(brain.regions) if i.lower().find('central')>-1]
side = 'L'
# v1=brain._generateAxis(1)
# v1=brain._plotBrainVolume(v1,.02,[1,1,1],side=side)
# v1=brain._plotBrainRegions(v1,regions=[i[0] for i in centralSulcusInfo], colors=[i[-1] for i in centralSulcusInfo],opacity=.15,side=side)

# v1.show()
a = SCAN_SingleSessionAnalysis(dataPath,subject,session,remove_trajectories=['OR'],
      load=loadData,plot_stimuli=False,gammaRange=gammaRange,refType=reref)
r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(saveMAT=save)
sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals,alpha=0.05)
effect_of_interest =r_sq
effect_name = 'r_sq'
datasubset = sig_chans
# datasubset = []
pp = True
if pp:            
      v1=brain._generateAxis(1)
      v1=brain._plotBrainVolume(v1,0.05,[1,1,1],side=side)
      v1=brain._plotBrainRegions(v1,regions=[i[0] for i in centralSulcusInfo], colors=[i[-1] for i in centralSulcusInfo],opacity=.2,side=side)
      intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(effect_of_interest,datasubset,save=save,label='significant',thresh=.1,comparison='')
      intereffectors = [i.replace('_','') for i in intereffectors]
      nonspecifics = [i.replace('_','') for i in nonspecifics]
      nonspecifics_locs = [electrodemap[i] for i in nonspecifics]
      intereffector_locs = [electrodemap[i] for i in intereffectors]
      print(channel_classifcation)
      print('intereffectors')
      for i,j in zip(intereffectors,intereffector_locs):
            print(i,': ',j)
      print('\n\nnonspecifics')
      non_specific= ['inter','foot-face','hand-face','hand-foot']
      multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
      targets = []
      for i,j in zip(multiMotor,nonspecifics_locs):
            if j.find('precentral')>-1 or j.find('S-central')>-1:
                  print(i[0],': ',j,' ',i[1])
                  targets.append(i[0])
      # targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
      # targetIdx.extend( brain._getElectrodeIndexFromLabels(labels=nonspecifics))
      targetIdx = brain._getElectrodeIndexFromLabels(labels=targets)
      v1=brain._plot_electrodes_on_volume(v1,electrodeSubset=targetIdx,side=side)
      v1.show()

print(0)
