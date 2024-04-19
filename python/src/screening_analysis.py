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
from stimulusPresentation import screening_session, format_Stimulus_Presentation_Session
from ERP_struct import ERP_struct
from SCAN_SingleSessionAnalysis import readPickle, writePickle, alphaSortDict

class screening_analysis(screening_session):
    def __init__(self,path:str or Path,subject:str,fs:int=2000,load=True,plot_stimuli:bool=False,gammaRange=[70,170],rerefSEEG='common') -> None: # type: ignore
        """
        Module containing functions for single session analysis of BCI2000 sensorimotor screening tasks

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
        """
        
        if type(path) == str:
                path = Path(path)
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
        self.sensorimotor.epoch_info = self.sensorimotor.epochStimulusCode_screening(plot_states=plot_stimuli)
        self.motor.epoch_info = self.motor.epochStimulusCode_screening(plot_states=plot_stimuli)
        self.sensory.epoch_info = self.sensory.epochStimulusCode_screening(plot_states=plot_stimuli)
        self.ERP_m_epochs = self._epochERPs(self.motor)
        self.ERP_s_epochs = self._epochERPs(self.sensory)
        self.ERP_sm_epochs = self._epochERPs(self.sensorimotor)
        self.motor.data = self._processSignals(self.motor,load=load,rerefSEEG=rerefSEEG)
        self.sensorimotor.data = self._processSignals(self.sensorimotor,load=load,rerefSEEG=rerefSEEG)
        self.sensory.data = self._processSignals(self.sensory,load=load,rerefSEEG=rerefSEEG)
        print('end init')  
    
    
    def extractERPs(self,recordingType:str,timeLag: float=2)-> dict:
        """This function extracts ERP epochs based on the type of recording specified (motor or sensory).
        
        Parameters
        ----------
        recordingType : str
            recordingType: a string indicating the type of recording for which ERPs are to be extracted. It
        can be either 'motor', 'sensory' or 'sm'.
        timeLag : float, optional
            The `timeLag` parameter specifies the time lag in seconds for epoching the data. This timeLag
        is applited both before and after the epoch interval. In this function, it is set to a default 
        value of 2 seconds if not provided by the user. 
        
        Returns:
            dict: dictionary of raw signal epochs
        """
        match recordingType:
            case 'motor':
                data = self.motor.data
                ERP_epochs = self._epochERPs(self.motor,timeLag=timeLag)
            case 'sensory':
                data = self.sensory.data
                ERP_epochs = self._epochERPs(self.sensory,timeLag=timeLag)
            case 'sm':
                data = self.sensorimotor.data
                ERP_epochs = self._epochERPs(self.sensorimotor,timeLag=timeLag)
        
        
        epochs = {}
        for k,v in ERP_epochs.items(): # epoch information (muscle, list of intervals)
            signals = {}
            for sigType, values in data.items(): # (type of signal, dictionary of all data)
                trajectories = {}
                for traj, chan in values.items(): # (name of specific trajectory, recording sites on the trajectory)
                    channel = {loc: [data[on:end+1] for (on, end) in v] for loc, data in chan.items()} # dict comprehension to build epochs from intervals on each channel on a trajectory
                    if sigType in signals:
                        signals[sigType].update(channel) 
                    else:
                        signals[sigType] = channel
                epochs[k] = signals
        gammaERP = self.highGamma_ERP(epochs)
        gammaERP.plotAverages_per_trajectory(self.subject)
        plt.show()
        return epochs
    
    
    
    def _epochERPs(self,session:format_Stimulus_Presentation_Session,timeLag:float=2)-> dict:
        '''This function is used to epoch event-related potentials (ERPs) from a given session with a
        specified time lag.
        
        Parameters
        ----------
        session : format_Stimulus_Presentation_Session
            The `session` parameter is of type `format_Stimulus_Presentation_Session`, which likely
        represents a session of stimulus presentation in a specific format. This parameter is required
        for the function `_epochERPs` to process and analyze the data related to the stimulus
        presentation session.
        timeLag : float, optional
            The `timeLag` parameter specifies the time lag in seconds for epoching the data. This timeLag
        is applited both before and after the epoch interval. In this function, it is set to a default 
        value of 2 seconds if not provided by the user.
        '''
        epoch_info = session.epoch_info
        epochs = {}
        timelag = self.fs * timeLag
        for i,j in epoch_info.items():
            if i.find('stop')<0:
                epochs[i] = [[p[0]-timelag,p[1]+timelag] for p in j]
        return epochs
    
    def highGamma_ERP(self,data,power: bool=True)-> ERP_struct:
        inputData = data
        output = self.general_ERP(inputData, self.gammaRange,power=power)
        print('gamma ERP run')
        return output
    def beta_ERP(self,data,power: bool=True)-> ERP_struct:
        inputData = data
        output = self.general_ERP(inputData, [14,30],power=power)
        return output
    def mu_ERP(self,data,power: bool=True)-> ERP_struct:
        inputData = data
        output = self.general_ERP(inputData, [12,15],power=power)
        return output
    def broadband_ERP(self,data,power: bool=False)-> ERP_struct:
        inputData = data
        output = self.general_ERP(inputData,power=power)
        return output
    def general_ERP(self,epochs,filterBand:list = [0.5,170],power:bool=False)->ERP_struct:
        """take epochs of data, overlay them per each channel and extract average ERPs
        ----------
        Parameters:
            epochs (dictionary): nested dict of signal types epoch
            filterBand (list, optional): lowcut and highcut for bandpass filtering. Defaults to [0.5,1000].
            power (bool, optional): true if power of the signal should be computed via the hilbert transform. Defaults to False.
        ----------
        Returns:
            ERP_struct: class with access to type, filterband, averages and raw data for an ERP,
            also 
        """
        print('starting ERP calc')
        aggregateData = {}
        for mv, data in epochs.items():
            averages = {}
            raw = {}
            stdevs = {}
            agg = data['sEEG']
            if 'EMG' in data:
                agg.update(data['EMG'])
            gen = ([channel,vals] for channel,vals in agg.items() if channel.find('REF')<0)
            for i in gen:
                channel = i[0]
                temp = []
                for epoch in i[1]:
                    if not any(char.isdigit() for char in channel):
                        "Parsing for EMG channels"
                        if channel[0:3] != 'EMG': 
                            channel = f'EMG_{channel}'       
                    elif power:
                        "Getting Timeseries Power"
                        epoch = bandpass(epoch,self.fs,filterBand,order=4)
                        epoch = hilbert_env(epoch)**2
                        epoch = zscore_normalize(epoch)
                        epoch = moving_average_np(epoch,1000)
                    else:
                        "Just looking at voltage"
                        epoch = bandpass(epoch,self.fs,filterBand,order=4)        
                    temp.append(epoch)
                averages[channel]=np.average(temp,axis=0)
                stdevs[channel] = np.std(temp,axis=0)
                raw[channel] = temp
            aggregateData[mv] = [averages,stdevs,raw]
        output = ERP_struct(aggregateData,filterBand,power,self.fs)
        return output
    
    
    def _segmentSignals(self,sigType:dict,dataObj:format_Stimulus_Presentation_Session):
        out = {}
        # The line `for k,i in self.channels.items():` is iterating over the items in the
        # `self.channels` dictionary. During each iteration, `k` represents the key of the current
        # item, and `i` represents the value associated with that key. This loop is used to process
        # and segment signals based on their type in the `screening_analysis` class.
        for k,i in dataObj.channels.items():
            if i == sigType:
                out[k] = dataObj.data[k]
        return out
    def _processSignals(self,dataObj:format_Stimulus_Presentation_Session,load=True,rerefSEEG=''):
        fname = f'{rerefSEEG}_processed.pkl'
        if fname in os.listdir(dataObj.root / 'preprocessed') and load==True:
            signalGroups = readPickle(dataObj.root / 'preprocessed' /fname)
            print('loaded prepocessed data')
        else:
            if rerefSEEG == 'common':
                        commonAVG = dataObj.getCommonAverages()
            else:
                commonAVG = None
            signalGroups = {}
            for sig in dataObj.signalTypes:
                signalGroups[sig] = self._segmentSignals(sig,dataObj)
            for sigType,data in signalGroups.items():
                if sigType == 'EMG':
                    data = self.processEMG(data)
                if sigType == 'sEEG': 
                    data = self.process_sEEG(data,rerefSEEG,commonAVG)
                if sigType == 'ECG':
                    data = self.processECG(data)
                if sigType == 'EEG':
                    data = self.processEEG(data)
                signalGroups[sigType] = data
            writePickle(signalGroups,dataObj.root / 'preprocessed',fname=fname)
            print('preprocessed the data')
        return signalGroups
    def processECG(self,ECG:dict):
        output ={}
        output['ECG'] = ECG
        return output
    def processEMG(self,EMG:dict,plotWorkFlow=False):
        muscles = set([i.split('_')[0] for i in EMG.keys()])
        hold = {}
        output = {}
        for muscle in muscles:
            data = [i for k,i in EMG.items() if k.find(muscle)>-1]
            data = self._bipolarReference(data[0],data[1])
            bp = bandpass(data,fs=self.fs,Wn=[25,400],order=3)
            n = notch(bp,self.fs,60,30,1)
            n = notch(n,self.fs,120,60,1)
            n = notch(n,self.fs,180,90,1)
            abs_n= np.abs(n)
            log10 = np.log10(abs_n)
            z = zscore_normalize(log10)
            # smoothz = savitzky_golay(z,window_size=int(self.fs/2)-1,order = 0)
            smoothz = moving_average_np(z,window_size=int(self.fs/2))
            # temp = hilbert_env(temp)
            expon_z = math.e**smoothz
            hold[muscle] = expon_z - 1
            if plotWorkFlow:
                fig,(a1,a2,a3,a4,a5,a6)  = plt.subplots(6,1, sharex=True)
                a1.plot(n)
                a1.set_ylabel('filt')
                a2.plot(abs_n)
                a2.set_ylabel('abs')
                a3.plot(log10)
                a3.set_ylabel('log10')
                a4.plot(z)
                a4.set_ylabel('z score')
                a5.plot(smoothz)
                a5.set_ylabel('smoothing')
                a6.plot(expon_z-1)
                a6.set_ylabel('exponentiated')
                plt.show()
            # hold[muscle] = temp
        output['EMG'] = hold
        return output
    def processEEG(self,EEG:dict):
        output = {}
        output['EEG'] = EEG
        return output
    def process_sEEG(self,sEEG:dict,reref:str='common',commonAV=None):
        trajectories = [key[0:2] for key in sEEG.keys()]
        trajectories = set(trajectories)
        trajectories.remove('RE')
        trajectories.add('REF')
        output = {}
        for traj in trajectories:
            match reref:
                case 'bipolar':
                    data = [v for k,v in sEEG.items() if k.find(traj)>-1]
                    traj_data = {}
                    for idx,vals in enumerate(data[0:-1]):
                        label = f'{traj}_{idx+1}_{idx+2}'
                        temp = self._bipolarReference(data[idx+1],vals)
                        # temp = notch(temp,self.fs,60,30,1)
                        traj_data[label] = temp 
                    output[traj] = traj_data
                
                case 'common':
                    
                    data = [v for k,v in sEEG.items() if k.find(traj)>-1]
                    traj_data = {}
                    for idx,vals in enumerate(data):
                        label = f'{traj}_{idx+1}'
                        temp = vals - commonAV['sEEG']
                        traj_data[label] = temp 
                    output[traj] = traj_data
                case _:
                    data = [v for k,v in sEEG.items() if k.find(traj)>-1]
                    traj_data = {}
                    for idx,vals in enumerate(data):
                        label = f'{traj}_{idx+1}'
                        temp = vals
                        # temp = notch(temp,self.fs,60,30,1)
                        traj_data[label] = temp 
                    output[traj] = traj_data
        output = alphaSortDict(output)
        return output
    def _bipolarReference(self,a,b):
        return b-a 
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
    a = screening_analysis(dataPath,subject,load=True,plot_stimuli=False,gammaRange=gammaRange,rerefSEEG='common')
    a.extractERPs('sensory')