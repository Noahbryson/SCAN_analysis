import os
import h5py
from pathlib import Path
import scipy.io as scio
import scipy.stats as stats
from functions.filters import *


class format_Stimulus_Presentation_Session():
    def __init__(self,loc:Path,subject,plot_stimuli=False,HDF:bool=False):
        """
        Formats the 4 preprocessed filed from MATLAB into a data object
        
        Parameters
        ----------
        loc: Path
            path to preprocessed directory
        subject: str
            individual subject ID
        """
        self.root = loc.parent
        files = os.listdir(loc)
        for file in files:
            if file.find(subject)>-1:
                if HDF:
                    # need to write HDF5 parser. 
                    # with h5py.File(loc/file, 'r') as f:
                    #     # data = {key: f[key][()] for key in f.keys()}
                    #     data = f['agg_signals'][()]
                    data = {}
                    hf = h5py.File(loc/file,'r')
                    temp = hf['agg_signals']
                    keys = list(temp.keys())
                    for k in keys:
                        data[k] = temp[k][0]
                    self.data = data
                    print(0)

                else:    
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
                print(f'{file} not loaded on init')
        self.epoch_info = self.epochStimulusCode_SCANtask(plot_stimuli)

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
    def epochStimulusCode_SCANtask(self,plot_states:False):
        data = self.states['StimulusCode']
        moveEpochs = {}
        onset_shift = 1000
        offset_shift = 3000
        for stim in self.stimuli.values(): # get intervals for each of the stimulus codes
            code = stim[0]['code']
            stim_type = stim[6]
            loc = np.where(data==code)
            intervals = find_intervals(loc[0])
            for i,v in enumerate(intervals):
                intervals[i] = [v[0]-onset_shift,v[1]+offset_shift]

            moveEpochs[stim_type] = intervals
        loc = np.where(data==0) # get intervals for stim code of zero (at rest)
        intervals = find_intervals(loc[0])
        for i,v in enumerate(intervals):
                intervals[i] = [offset_shift+v[0],v[1]-onset_shift]
        restEpochs = {}
        for k, int_set in moveEpochs.items():
            temp = []
            for i in int_set:
                onset = i[1]+1
                temp.append(extractInterval(intervals,onset))
            mode_int = stats.mode([i[1]-i[0] for i in temp]).mode
            for i in temp:
                if (i[1]-i[0]) != mode_int:
                    i[1] = i[0] + mode_int

            restEpochs[k] = temp                

        if plot_states:
            self.plotStimuli(moveEpochs)
        return moveEpochs, restEpochs
    def _reshapeStates(self):
        states = {}
        for state, val in self.states.items():
            mode = stats.mode(val).count
            if len(val) == mode:
                pass # Exclude states that do not contain any information, ie array contains all of one value. 
            else:
                states[state] = val
        return states
    def plotStimuli(self,epochs):
        data = self.states['StimulusCode']
        t = np.linspace(0,len(data)/2000,len(data))
        fig=plt.figure()
        ax = plt.subplot(1,1,1)
        ax.plot(t,data)
        for i in epochs.values():
            for j in i:
                ax.axvline(t[j[0]], c=(0,1,0))
                ax.axvline(t[j[1]], c=(1,0,0))
        scio.savemat(self.root/'analyzed'/'stimcode.mat',{'stim':data})
        scio.savemat(self.root/'analyzed'/'stimuli.mat',epochs)
        plt.show()
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
    
    # Add the last interval if the array does not end on a jump based on length of previous intervals
    int_length = intervals[-1][1] - intervals[-1][0]
    intervals.append((start, start+int_length))

    return intervals

def extractInterval(intervals,b):
    for i in intervals:
        if i[0] ==b: 
            return i
    return None

def sliceArray(array, interval):
    return array[interval[0]:interval[1]]