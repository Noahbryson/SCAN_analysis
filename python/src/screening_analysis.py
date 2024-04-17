import os
import csv
from pathlib import Path
import scipy.io as scio
import pandas as pd
from functions.filters import *
import math
import pickle
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from functions.stat_methods import mannwhitneyU, cohendsD, calc_ROC, euclidean_distance
from stimulusPresentation import screening_session
from ERP_struct import ERP_struct

class screening_analysis(screening_session):
      def __init__(self,path:str or Path,subject:str,fs:int=2000,load=True,plot_stimuli:bool=False,gammaRange=[70,170]) -> None: # type: ignore
            """
            Module containing functions for single session analysis of BCI2000 screening tasks

            Parameters
            ----------
            path: str or Path
            path to SCAN experiment data repository for each participant.
            subject: str
            subject ID
            session: str
            name of the session targeted for analysis. typical values are pre_ablation and post_ablation
            fs: int
            sampling frequency of the data, default is 2000 -> will deprecate and automate in the future.
            load: bool
            load saved datastructure of preprocessed data to speed up run-time. set to false if reprocessing is needed.
            see self._processSignals
            determine wheter or not movement trials are epoched via EMG onset of via stimulus onset.
            """
            if type(path) == str:
                  path = Path(path)
            print(f'')
            self.main_dir = path
            self.subject = subject
            self.fs = fs
            
            if os.path.exists(self.main_dir/self.subject/'muscle_mapping.csv'):
                  with open(self.main_dir/self.subject/'muscle_mapping.csv', 'r') as fp:
                        reader = csv.reader(fp)
                        self.muscleMapping = {rows[0]:rows[1:] for rows in reader}
            else:
                  self.muscleMapping = {'1_Hand':['wristExtensor', 'ulnar'], '3_Foot':['TBA'],'2_Tongue':['tongue']}
            self.subjectDir = path / subject
            super().__init__(self.subjectDir,subject,plot_stimuli=plot_stimuli,HDF=True)
            self.saveRoot = self.subjectDir / 'analyzed'
            self.gammaRange = gammaRange
            print('end init')
            

if __name__ == '__main__':
    import platform
    localEnv = platform.system()
    userPath = Path(os.path.expanduser('~'))
    if localEnv == 'Windows':
        dataPath = userPath / r"Box\Brunner Lab\DATA\SCREENING"
    else:
        dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCREENING"
    subject = 'BJH041'
    gammaRange = [70,170]
    a = screening_analysis(dataPath,subject,load=True,plot_stimuli=False,gammaRange=gammaRange)