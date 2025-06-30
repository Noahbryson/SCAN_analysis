import os
from pathlib import Path
from src.screening_analysis import *
import matplotlib.pyplot as plt
import sys
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain
import platform
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCREENING"
else:
      dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCREENING"

targetRegions = ['AV','LP','CM','MD','VL','PU','PuM','PuL','PuI','LGN',
                  'MGN','PuA','CL','VM','LP','LD','VA','central']



subject = 'BJH069'
gammaRange = [70,170]
import datetime
print(f'\n\n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n\n')

# """Testing Block"""
# xxx = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"/'BJH058'/'brain'/'brain_new.mat'
# brain = PyBrain(fp=xxx,subject=subject,brainName='MNI')
# """End Testing Block"""


bipolarFlag = False
brain = PyBrain(fp=dataPath/subject/'brain/brain_MNI.mat',subject=subject,brainName='MNI')
if bipolarFlag:
      reref = 'bipolar'
      brain._makeBipolarElectrodes()
else:
      reref = 'common'
print(subject)

for i in sorted(set(brain.regions)):
      ii = str.replace(i,'ctx-','')
      print(ii)

# v1 = brain._generateAxis(1)
# v1=brain._plotBrainVolume(v1,0.05,color=[1,1,1])
# v1=brain._ColorBrainRegion(v1,)

a = screening_analysis(dataPath,subject,load=True,plot_stimuli=False,gammaRange=gammaRange,rerefSEEG=reref)
a.compareTaskSpectra('sm',save=True)
# a.compareTaskSpectra('sensory',save=True)
# a.extractERPs('sensory')
# a.extractERPs('sm',save=True)
sensationResponse = a.extractSensoryResponses(save=dataPath/subject/'analyzed')
square = int(np.ceil(np.sqrt(len(sensationResponse))))
vol = brain._generateMultiAxis(square,square,link=True)
row, col = 0, 0
for i,k in enumerate(sensationResponse):
      if col >= square:
            row = row + 1
            col = 0
      vol.subplot(row,col)
      vol = brain._plotBrainVolume(vol,0.05,color=[1,1,1],side='l')
      brain._plotEffectOnVolume(vol,sensationResponse[k],significant=True,electrodeSubset=[])
      vol.add_title(k)
      col = col +1 

# vol = brain._generateAxis()
# vol = brain._plotBrainVolume(vol,0.05,color=[1,1,1])
# brain._plotEffectOnVolume(vol,sensationResponse['index'],significant=True)
      



vol.show()
print(0)