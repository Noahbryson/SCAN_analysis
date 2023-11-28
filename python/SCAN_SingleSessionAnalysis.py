import os
from pathlib import Path
import scipy.io as scio
import scipy.stats as st
from filters import *
import math
import time
import pickle

class SCAN_SingleSessionAnalysis():
    def __init__(self,path:str or Path,subject:str,session:str,fs:int=2000,load=True) -> None:
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
        self.muscleMapping = {'1_Hand':['wristExtensor', 'ulnar'], '3_Foot':['TBA'],'2_Tongue':['tongue']}
        self.subjectDir = path / subject / session
        dataLoc = self.subjectDir / 'preprocessed'
        self.session_data = format_Stimulus_Presentation_Session(dataLoc,subject)
        self.signalTypes = set(self.session_data.channels.values())
        
        self.session_data.data = self._processSignals(load)
        self.move_epochs = self._epochData('move')
        self.rest_epochs = self._epochData('rest')
        self._alignByMovementOnset()
        print('end init')


    def _processSignals(self,load=True):
        if 'processed.pkl' in os.listdir(self.subjectDir / 'preprocessed') and load==True:
            signalGroups = readPickle(self.subjectDir / 'preprocessed' /'processed.pkl')
        else:
            signalGroups = {}
            for sig in self.signalTypes:
                signalGroups[sig] = self._segmentSignals(sig)
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
            writePickle(signalGroups,self.subjectDir / 'preprocessed')
        return signalGroups
    def processECG(self,ECG:dict):
        output ={}
        output['ECG'] = ECG
        return output
    def processEMG(self,EMG:dict):
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
            log10= np.abs(n)
            log10 = np.log10(log10)
            z = zscore_normalize(log10)
            # smoothz = savitzky_golay(z,window_size=int(self.fs/2)-1,order = 0)
            smoothz = moving_average_np(z,window_size=int(self.fs/2))
            # temp = hilbert_env(temp)
            expon_z = math.e**smoothz
            hold[muscle] = expon_z - 1
            # hold[muscle] = temp
        output['EMG'] = hold
        return output
    def processEEG(self,EEG:dict):
        output = {}
        output['EEG'] = EEG
        return output
    def process_sEEG(self,sEEG:dict):
        trajectories = [key[0:2] for key in sEEG.keys()]
        trajectories = set(trajectories)
        trajectories.remove('RE')
        trajectories.add('REF')
        bandSplit = ([65,75],[75,85],[85,95],[95,105],[105,115])
        output = {}
        for traj in trajectories:
            data = [v for k,v in sEEG.items() if k.find(traj)>-1]
            traj_data = {}
            for idx,vals in enumerate(data[0:-1]):
                label = f'{traj}_{idx+1}-{idx+2}'
                temp = self._bipolarReference(data[idx+1],vals)
                gamma = getGammaBand_sEEG(temp,self.fs,order=3)
                bands = np.empty([len(bandSplit),len(gamma)])
                for i,band in enumerate(bandSplit):
                    p = bandpass(gamma,self.fs,Wn=band,order=3)
                    pxx = hilbert_env(p) **2
                    bands[i] = pxx
                
                temp = sum(bands)
                # ax = plt.subplot(6,1,1)
                # ax.plot(temp)
                temp = np.log10(temp)
                # ax = plt.subplot(6,1,2)
                # ax.plot(temp)
                temp = zscore_normalize(temp)
                # ax = plt.subplot(6,1,3)
                # ax.plot(temp)
                temp = moving_average_np(temp,1000)
                # temp = savitzky_golay(temp,window_size=999,order=0)
                
                # ax = plt.subplot(6,1,4)
                # ax.plot(temp)
                temp = math.e**temp
                # ax = plt.subplot(6,1,5)
                # ax.plot(temp)
                temp = temp - 1
                # ax = plt.subplot(6,1,6)
                # ax.plot(temp)
                traj_data[label] = temp 
            output[traj] = traj_data
        output = alphaSortDict(output)
        return output
    
    def _segmentSignals(self,sigType:dict):
        out = {}
        for k,i in self.session_data.channels.items():
            if i == sigType:
                out[k] = self.session_data.data[k]
        return out
    def _bipolarReference(self,a,b):
        return b-a  


    def _epochData(self,cond:str):
        if cond == 'move':
            idx = 0
        if cond == 'rest':
            idx = 1
        epochs = {}
        for muscle, intervals in self.session_data.epoch_info[idx].items(): # epoch information (muscle, list of intervals)
            signals = {}
            for sigType, values in self.session_data.data.items(): # (type of signal, dictionary of all data)
                trajectories = {}
                for traj, chan in values.items(): # (name of specific trajectory, recording sites on the trajectory)
                    channel = {loc: [data[on:end+1] for (on, end) in intervals] for loc, data in chan.items()} # dict comprehension to build epochs from intervals on each channel on a trajectory
                    trajectories[traj] = channel
                signals[sigType] = trajectories  
            epochs[muscle] = signals
        return epochs

    def _locateMuscleOnset(self,emg_stream,testplot=False):
        thresh = 0.5 *np.std(emg_stream)
        # grad = np.gradient(emg_stream,1/self.fs)
        # grad = savitzky_golay(grad,251,1)
        # grad = grad / max(abs(emg_stream))
        # deriv = np.diff(emg_stream, n=1)
        peaks_cwt = sig.find_peaks_cwt(emg_stream, widths = 500, noise_perc=thresh)
        peak_thresh = .1*emg_stream[peaks_cwt[0]]
        onset = peaks_cwt[0]
        while onset > 0 and emg_stream[onset] > peak_thresh:
            onset -= 1
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        # ax.plot(grad, label='grad')
        ax.plot(emg_stream, label='data')
        # ax.plot(deriv, label='deriv')
        ax.axhline(thresh, label='thresh',c=(0,0,0))
        ax.axvline(onset, c=(0,0,0))
        for peak in peaks_cwt:
            ax.axvline(peak, c=(1,0,0),label='_')
        ax.legend()
        plt.show()
        return onset
    def _reshapeEpoch(self,onset, epochData):
        pass

    def _alignByMovementOnset(self):
        output = {}
        for m_type, data in self.move_epochs.items():
            if m_type.find('rest') <0:
                epochOnsets = []
                emg = {x:data['EMG']['EMG'][x] for x in self.muscleMapping[m_type]}
                keys = list(emg.keys())
                numEpochs = len(emg[keys[0]])
                for i in range(numEpochs):
                    onset = 1e10
                    for muscle in self.muscleMapping[m_type]:
                        dat = emg[muscle][i]
                        temp = self._locateMuscleOnset(dat,testplot=True)
                        if temp < onset:
                            onset = temp
                    epochOnsets.append(onset)
                output[m_type] = epochOnsets
                
        return output
class format_Stimulus_Presentation_Session():
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
                self.states = self._reshapeStates()
                
            elif file.find('stimuli')>-1:
                stimuli = scio.loadmat(loc/file,mat_dtype=True,simplify_cells=True)
                stimuli = stimuli['stim_codes']
                self.stimuli = self.reshapeStimuliMatrix(stimuli=stimuli)
            else:
                print(f'{file} not loaded')
        self.epoch_info = self.epochStimulusCode()

    def reshapeStimuliMatrix(self,stimuli):
        keys = stimuli[0].keys()
        output = {}
        for value,key in enumerate(keys):
            temp = []
            for entry in stimuli:
                temp.append(entry[key])
            temp.insert(0,{'code':value+1})
            output[key] = temp
        return output
    def epochStimulusCode(self):
        data = self.states['StimulusCode']
        moveEpochs = {}
        shift = 1000
        for stim in self.stimuli.values(): # get intervals for each of the stimuls codes
            code = stim[0]['code']
            stim_type = stim[6]
            loc = np.where(data==code)
            intervals = find_intervals(loc[0])
            for i,v in enumerate(intervals):
                intervals[i] = [v[0]-shift,v[1]]

            moveEpochs[stim_type] = intervals
        loc = np.where(data==0) # get intervals for stim code of zero (at rest)
        intervals = find_intervals(loc[0])
        for i,v in enumerate(intervals):
                intervals[i] = [v[0],v[1]-shift]
        restEpochs = {}
        for k, int_set in moveEpochs.items():
            temp = []
            for i in int_set:
                onset = i[1]+1
                temp.append(extractInterval(intervals,onset))
            restEpochs[k] = temp                

        # epochs['rest'] = intervals[1:]
        return moveEpochs, restEpochs
    def _reshapeStates(self):
        states = {}
        for state, val in self.states.items():
            mode = st.mode(val).count
            if len(val) == mode:
                pass # Exclude states that do not contain any information, ie array contains all of one value. 
            else:
                states[state] = val
        return states


def alphaSortDict(a:dict)->dict:
    sortkeys = sorted(a)
    output = {k:a[k] for k in sortkeys}
    return output

def find_intervals(array):
    # Initialize the list of intervals
    intervals = []
    # Start the current interval with the first element
    start = array[0]

    # Iterate over the array starting from the second element
    for i in range(1, len(array)):
        # If the difference from the previous to the current is not 1, we've found the end of an interval
        if array[i] - array[i - 1] != 1:
            # Add the (start, end) of the interval to the list
            intervals.append((start, array[i - 1]))
            # Start a new interval with the current element
            start = array[i]
    
    # Add the last interval if the array does not end on a jump
    if len(array) > 1 and array[-1] - array[-2] == 1:
        intervals.append((start, array[-1]))

    return intervals

def writePickle(struct,fpath:Path):
    fname = fpath / 'processed.pkl'
    with open(fname,'wb') as handle:
        pickle.dump(struct,handle)
def readPickle(fpath):
    with open(fpath, 'rb') as handle:
        out = pickle.load(handle)
    return out

def extractInterval(intervals,b):
    for i in intervals:
        if i[0] ==b: 
            return i
    return None