import os
from pathlib import Path
from src.screening_analysis import *
import matplotlib.pyplot as plt
import sys
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain
import platform
import pandas as pd
import numpy as np

localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCREENING"
else:
      dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCREENING"
     
subject = 'BJH069'

targetRegions = ['AV','LP','CM','MD','VL','PU','PuM','PuL','PuI','LGN',
                  'MGN','PuA','CL','VM','LP','LD','VA','central']

side = 'both'


import datetime
print(f'\n\n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n\n')
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
thalamElec, thalamIdx = brain._isolateTargetElectrodes(['AV','LP','CM','MD','VL','PU','PuM','PuL','PuI','LGN',
                  'MGN','PuA','CL','VM','LP','LD','VA','central'])
print(thalamElec)

# df = pd.read_csv(dataPath/subject/'analyzed'/'agg_sensory_responses.csv')
df = pd.read_csv(dataPath/subject/'analyzed'/'sm'/'power'/'sm_powerRes.csv')

metric = 'rsq'
df.rename(columns={metric:'metric'},inplace=True)
# responses = {}
# for i in set(df['task']):
#       responses[i] = df[['channel','metric','p']]
square = int(np.ceil(np.sqrt(len(set(df['task'])))))
vol = brain._generateMultiAxis(square,square,link=True)
row, col = 0, 0
for i,k in enumerate(sorted(set(df['task']))):
      data = df[['channel','metric','p']].loc[df['task']== k]
      if col >= square:
            row = row + 1
            col = 0
      vol.subplot(row,col)
      vol = brain._plotBrainVolume(vol,0.05,color=[1,1,1],side=side)
      brain._plotEffectOnVolume(vol,data,significant=True,electrodeSubset=[],side=side)
      vol.add_title(k)
      col = col +1 
vol.show(auto_close=False,interactive=True)

# input('run complete? [y|n]:')

# print('run complete')