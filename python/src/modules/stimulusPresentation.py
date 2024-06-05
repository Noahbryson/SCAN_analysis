import os
import h5py
from pathlib import Path
import scipy.io as scio
import scipy.stats as stats
from functions.filters import *
import pickle
import pandas as pd 
import re
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
        self.name = 'stim_presentation'
        self.epoch_info = {} # intialize empty as different experiments have different requirements for this variable.
        self.task_epochs = {}
        self.rest_epochs = {}
        for file in files:
            if file.find(subject)>-1:
                if HDF:
                    # need to write HDF5 parser. 
                    # with h5py.File(loc/file, 'r') as f:
                    #     # data = {key: f[key][()] for key in f.keys()}
                    #     data = f['agg_signals'][()]
                    data = {}
                    hf = h5py.File(loc/file,'r')
                    try:
                        temp = hf['agg_signals']
                    except KeyError:
                        temp = hf['signals']
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
                self.stimuli = self._reshapeStimuliMatrix(stimuli=stimuli)
            else:
                print(f'{file} not loaded on init')
        temp = {k:self.channels[k] for k in self.data.keys()}
        self.channels = temp
        self.signalTypes = set(self.channels.values())
        # if os.path.exists(self.main_dir/self.subject/'muscle_mapping.csv'):
        #         with open(self.main_dir/self.subject/'muscle_mapping.csv', 'r') as fp:
        #             reader = csv.reader(fp)
        #             self.muscleMapping = {rows[0]:rows[1:] for rows in reader}
        # else:
        #         self.muscleMapping = {'1_Hand':['wristExtensor', 'ulnar'], '3_Foot':['TBA'],'2_Tongue':['tongue']}

    def getCommonAverages(self):
        """
        The function `getCommonAverages` calculates the average values for each signal type across all
        channels.
        :return: The `getCommonAverages` method returns a dictionary `common_avg` where the keys are
        signal types from `self.signalTypes` and the values are the mean values of the data for each
        signal type across all channels.
        """
        common_avg = {}
        for sigtype in self.signalTypes:
            temp = []
            for channel,t in self.channels.items():
                if t==sigtype:
                    temp.append(self.data[channel])
            common_avg[sigtype] = np.mean(temp,axis=0)
        return common_avg
    def _reshapeStimuliMatrix(self,stimuli):
        keys = stimuli[0].keys()
        output = {}
        for value,key in enumerate(keys):
            temp = []
            for entry in stimuli:
                temp.append(entry[key])
            temp.insert(0,{'code':value+1})
            output[key] = temp
        return output
    def _saveSelf(self,loc: Path):
        # TODO: make these modules save and load themselves from files to speed up operations, especially since this operation does not alter the data in any way.
        # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
        fname = loc/'session_aggregate.pkl'
        with open(fname, 'wb') as fp:
            pickle.dump(self)

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
    
    def epochStimulusCode_screening(self,plot_states:False):
        """
        This function extracts epochs of stimulus codes from the data and categorizes them into task and
        rest epochs for further analysis.
        
        :param plot_states: The `plot_states` parameter in the `epochStimulusCode_screening` function is
        a boolean parameter that determines whether to plot the stimuli epochs or not. If `plot_states`
        is set to `True`, the function will call the `plotStimuli` method to visualize the epochs
        :type plot_states: False
        :return: The function `epochStimulusCode_screening` returns the dictionary `epochs`, which
        contains intervals for each stimulus code.
        """
        data = self.states['StimulusCode']
        epochs = {}
        onset_shift = 0
        offset_shift = 0
        for stim in self.stimuli.values(): # get intervals for each of the stimulus codes
            code = stim[0]['code']
            stim_type = f'{stim[1]}_{code}'
            loc = np.where(data==code)
            intervals = find_intervals(loc[0])
            # if len(np.shape(intervals))==1:
            #     intervals[0] = intervals[0]-onset_shift
            #     intervals[1] = intervals[1]+offset_shift
            # else:
            #     for i,v in enumerate(intervals):
            #         intervals[i] = [v[0]-onset_shift,v[1]+offset_shift]
            epochs[stim_type] = intervals
        if plot_states:
            self.plotStimuli(epochs)    
        item_keys = set([i.split('_')[0] for i in epochs.keys()])
        output = {k:[] for k in item_keys}
        for k,v in epochs.items():
            kt = k.split('_')[0]
            if kt in item_keys:
                output[kt].append(v)
        restInts = epochs['stop and relax_1']
        restOnsets = np.array([i[0] for i in restInts],dtype=int)
        taskEpochs = {}
        restEpochs = {}
        for k,v in output.items():
            if k.find('stop')<0:
                if len(v) == 1:
                    v = v[0]
                temp = []
                for i in v:
                    loc = getNearestGreaterValueLoc(i[-1],restOnsets)
                    temp.append(restInts[loc])
                restEpochs[k] = temp
                taskEpochs[k] = v
        self.task_epochs = taskEpochs
        self.rest_epochs = restEpochs               
        return epochs
    
    def orderedEpochDf(self,task:dict,rest:dict):
        cols = ['task','taskInt','restInt','trial']
        temp = []
        for t,r in zip(task.keys(),rest.keys()):
            for i,(j,k) in enumerate(zip(task[t],rest[r])):
                temp.append([t,j,k,i+1])
        out = pd.DataFrame(temp,columns=cols)
        return out
    
    
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
        import distinctipy
        data = self.states['StimulusCode']
        t = np.linspace(0,len(data)/2000,len(data))
        fig=plt.figure()
        ax = plt.subplot(1,1,1)
        ax.plot(t,data)
        cmap = distinctipy.get_colors(len(epochs))
        for i,k in enumerate(epochs.keys()):
            # vals = int(k.split('_')[0])
            val = int(re.findall(r'\d+', k)[0])
            for q,j in enumerate(epochs[k]):
                if q == 0:
                    # ax.axvline(t[j[0]], c=cmap[i], label=k)
                    # ax.axvline(t[j[1]], c=(0,0,0),label='_')
                    ax.plot(t[j[0]:j[1]],val*np.ones(len(t[j[0]:j[1]])),c=cmap[i], label=k)
                else:
                    # ax.axvline(t[j[0]], c=cmap[i],label='_')
                    # ax.axvline(t[j[1]], c=(0,0,0),label='_')
                    ax.plot(t[j[0]:j[1]],val*np.ones(len(t[j[0]:j[1]])),c=cmap[i], label="_")
        # scio.savemat(self.root/'analyzed'/'stimcode.mat',{'stim':data})
        # scio.savemat(self.root/'analyzed'/'stimuli.mat',epochs)
        ax.set_ylim(-1,max(data)+2)
        ax.legend()
        ax.set_title(self.name)
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
    if len(intervals) > 0:
        int_length = intervals[-1][1] - intervals[-1][0]
        intervals.append([start, start+int_length])
    else:
        intervals = [array[0],array[-1]]
    return intervals


class screening_session(object):
    """docstring for screening_session."""
    # TODO: make these modules save and load themselves from files to speed up operations, especially since this operation does not alter the data in any way.
    # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
        
    def __init__(self, loc:Path,subject:str,plot_stimuli: bool=False, HDF:bool=True):
        super(screening_session, self).__init__()
        self.motor =        format_Stimulus_Presentation_Session(loc/'motor/preprocessed',subject,plot_stimuli=plot_stimuli,HDF=HDF)
        self.motor.name = 'motor'
        self.sensory =      format_Stimulus_Presentation_Session(loc/'sensory/preprocessed',subject,plot_stimuli=plot_stimuli,HDF=HDF)
        self.sensory.name = 'sensory'
        self.sensorimotor = format_Stimulus_Presentation_Session(loc/'sensory-motor/preprocessed',subject,plot_stimuli=plot_stimuli,HDF=HDF)
        self.sensorimotor.name = 'sensorimotor'

def extractInterval(intervals,b):
    for i in intervals:
        if i[0] ==b: 
            return i
    return None

def sliceArray(array, interval):
    return array[interval[0]:interval[1]]

def getNearestGreaterValueLoc(value, b:np.ndarray):
    # a = b[np.where(b>value)]
    # output = min(a,key=lambda x:(x-value))
    # outputIdx, = np.where(a==output)
    outputIdx, = np.where((b-value)==1)
    return outputIdx[0]