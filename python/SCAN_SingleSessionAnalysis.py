import os
from pathlib import Path
import scipy.io as scio
from filters import *
import math


class SCAN_SingleSessionAnalysis():
    def __init__(self,path:str or Path,subject:str,session:str,fs:int=2000) -> None:
        """
        Module containing functions for single session analysis of BCI2000 SCAN task

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
        """
        if type(path) == str:
            path = Path(path)
        self.main_dir = path
        self.subject = subject
        self.session = session
        self.fs = fs
        self.subjectDir = path / subject / session
        dataLoc = self.subjectDir / 'preprocessed'
        self.session_data = formatSession(dataLoc,subject)
        self.signalTypes = set(self.session_data.channels.values())
        self.processSignals()
        print('end init')


    def processSignals(self):
        signalGroups = {}
        for sig in self.signalTypes:
            signalGroups[sig] = self.segmentSignals(sig)
        for sigType,data in signalGroups.items():
            if sigType == 'EMG':
                data = self.processEMG(data)
            if sigType == 'sEEG':
                data = self.process_sEEG(data)
            if sigType == 'ECG':
                data = self.processECG(data)
            if sigType == 'EEG':
                data = self.processEEG(data)
            signalGroups[sigType] = data
        return signalGroups


    def processECG(self,ECG:dict):
        pass
    def processEMG(self,EMG:dict):
        muscles = set([i.split('_')[0] for i in EMG.keys()])
        output = {}
        for muscle in muscles:
            data = [i for k,i in EMG.items() if k.find(muscle)>-1]
            data = self.bipolarReference(data[0],data[1])
            temp = bandpass(data,fs=self.fs,Wn=[25,400],order=3)
            temp = notch(temp,self.fs,60,30,1)
            temp = notch(temp,self.fs,120,60,1)
            temp = notch(temp,self.fs,180,90,1)
            temp= np.abs(temp)
            temp = np.log10(temp)
            temp = zscore_normalize(temp)
            temp = savitzky_golay(temp,window_size=int(self.fs/2)-1,order = 0)
            # temp = hilbert_env(temp)
            temp = math.e**temp
            output[muscle] = temp
        return output
    def processEEG(self,EEG:dict):
        return EEG
    def process_sEEG(self,sEEG:dict):
        pass
    
    def segmentSignals(self,sigType:dict):
        out = {}
        for k,i in self.session_data.channels.items():
            if i == sigType:
                out[k] = self.session_data.data[k]
        return out
    def bipolarReference(self,a,b):
        return a-b  

class formatSession():
    def __init__(self,loc:Path,subject):
        """
        Formats the 4 preprocessed filed from MATLAB into a data object
        
        Parameters
        ----------
        loc: Path
            path to preprocessed directory
        subject: str
            individual subject ID
        """
        files = os.listdir(loc)
        for file in files:
            if file.find(subject)>-1:
                data = scio.loadmat(loc/file,mat_dtype=True,simplify_cells=True)
                self.data = data['signals']
            elif file.find('channeltypes')>-1:
                channels = scio.loadmat(loc/file,mat_dtype=True,simplify_cells=True)
                self.channels = channels['chan_types']
            elif file.find('states')>-1:
                states = scio.loadmat(loc/file,mat_dtype=True,simplify_cells=True)
                self.states = states['states']
            elif file.find('stimuli')>-1:
                stimuli = scio.loadmat(loc/file,mat_dtype=True,simplify_cells=True)
                stimuli = stimuli['stim_codes']
                self.stimuli = self.reshapeStimuliMatrix(stimuli=stimuli)
            else:
                print(f'{file} not loaded')

    def reshapeStimuliMatrix(self,stimuli):
        keys = stimuli[0].keys()
        output = {}
        for value,key in enumerate(keys):
            temp = []
            for entry in stimuli:
                temp.append(entry[key])
            temp.insert(0,{'code':value})
            output[key] = temp
        return output