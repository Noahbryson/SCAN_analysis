import os
from pathlib import Path
import scipy.io as scio
import scipy.stats as stats
import pandas as pd
from filters import *
import math
import time
import pickle
from sklearn import metrics
from stat_methods import mannwhitneyU, cohendsD

class SCAN_SingleSessionAnalysis():
    def __init__(self,path:str or Path,subject:str,session:str,fs:int=2000,load=True,epoch_by_movement:bool=True,plot_stimuli:bool=False) -> None:
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
        load: bool
            load saved datastructure of preprocessed data to speed up run-time. set to false if reprocessing is needed.
            see self._processSignals
        epoch_by_movement: bool, default = True
            determine wheter or not movement trials are epoched via EMG onset of via stimulus onset.
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
        self.saveDir = self.subjectDir / 'analyzed'
        self.session_data = format_Stimulus_Presentation_Session(dataLoc,subject,plot_stimuli=plot_stimuli)
        self.signalTypes = set(self.session_data.channels.values())
        
        self.session_data.data = self._processSignals(load)
        self.sessionEMG = self.session_data.data['EMG']
        self.move_epochs = self._epochData('move')
        self.rest_epochs = self._epochData('rest')
        self.motor_onset = self._EMG_activity_epochs(testplots=False)
        self.move_epochs,self.rest_epochs = self.reshape_epochs()
        if epoch_by_movement:
            self.move_epochs = self._epoch_via_EMG()
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
    def process_sEEG(self,sEEG:dict,gamma_power:bool=False):
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
                label = f'{traj}_{idx+1}_{idx+2}'
                temp = self._bipolarReference(data[idx+1],vals)
                # temp = notch(temp,self.fs,60,30,1)
                if gamma_power:
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
    def export_epochs(self,signalType,fname):
        out = {}
        for i in list(self.muscleMapping.keys()):
            move = self.move_epochs.query("type==@signalType & movement==@i")
            numCols = [i for i in move.columns if type(i)==int]
            for row in move.iterrows():
                temp = row[1][numCols].to_numpy()
                # temp.insert(0, row[1]['movement'])
                out[row[1]['name']] = temp
            dir = self.saveDir
            scio.savemat(dir/f'{fname}_{i}.mat',out)
        return 0
    def export_session_EMG(self):
        dat = self.sessionEMG['EMG']
        scio.savemat(self.saveDir/'fullEMG.mat',dat)

    def reshape_epochs(self):
        move = pd.DataFrame()
        for k,d in self.move_epochs.items():
            move = pd.concat([move,self.epochs_to_df(d,k)])
        rest = pd.DataFrame()
        for k,d in self.rest_epochs.items():
            rest = pd.concat([rest,self.epochs_to_df(d,k)])
        return move,rest
        
    def epochs_to_df(self,target:dict,ID:str):
        """used for reformatting epoch dicts into dataframes where columns are each epoch and rows are each channel
            the first column of the df will be a description of channel types for easier parsing
            ----------
            Params
            ----------
            target: dict
                nested dictionary structure containing epoch data for one movement type
                will work on rest and movement datastructures
            ID: str
                name of movement condition being fed in. 
            """
        remap = {}
        for k,i in target.items():
            temp = {}
            for j in i.values():
                temp.update(j)
            remap[k] = temp
        big_guy = []
        for c_type,v in remap.items():
            for a,b in v.items():
                    temp = [a,c_type,ID]
                    temp.extend(b)
                    big_guy.append(temp)
        num_epochs = len(big_guy[0]) - 3
        headers = [i+1 for i in range(num_epochs)];headers.insert(0,'movement'),headers.insert(0,'type');headers.insert(0,'name')
        return pd.DataFrame(big_guy,columns=headers)
        
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
        peak_thresh = .08*emg_stream[peaks_cwt[0]]
        onset = peaks_cwt[0]
        while onset > 0 and emg_stream[onset] > peak_thresh:
            onset -= 1
        start = int(onset - 0.5*self.fs) # shift 500ms in time to get window before movement begins
        if start < 0:
            start = 0
        stop = int((4 * self.fs) + onset) # step to 4s after movement onset
        # therefor range of start:stop should be 4.5*fs, in nihon-kohden case is 9000 samples
        if testplot:
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            # ax.plot(grad, label='grad')
            ax.plot(emg_stream, label='data')
            # ax.plot(deriv, label='deriv')
            ax.axhline(thresh, label='thresh',c=(0,0,0))
            ax.axvline(onset, c=(0,0,1))
            ax.axvline(onset-1000, c=(0,0,1),alpha=0.6)
            ax.axvline(onset+4.5*self.fs,c=(0,0,1))
            ax.axvline(onset-1000+4.5*self.fs,c=(0,0,1),alpha=0.6)
            for peak in peaks_cwt:
                ax.axvline(peak, c=(1,0,0),label='_')
                
            ax.legend()
            plt.show()
        return [start,stop]
    def _epoch_via_EMG(self):
        if type(self.move_epochs) != pd.DataFrame:
            self.move_epochs, self.rest_epochs = self.reshape_epochs()
        output = pd.DataFrame()
        for m, ints in self.motor_onset.items():
            df = self.move_epochs.query("movement==@m")
            strKeys = df.columns.to_list()
            strKeys = [i for i in strKeys if type(i)==str]
            temp = df[strKeys].copy()
            for i,j in enumerate(ints):
                temp[i+1] = df[i+1].apply(lambda x:sliceArray(x,j)) 
            
            output = pd.concat([output,temp])
            del temp
        return output
        

    def _EMG_activity_epochs(self,testplots=False):
        output = {}
        for m_type, data in self.move_epochs.items():
            if m_type.find('rest') <0:
                epochOnsets = []
                emg = {x:data['EMG']['EMG'][x] for x in self.muscleMapping[m_type]}
                keys = list(emg.keys())
                numEpochs = len(emg[keys[0]])
                for i in range(numEpochs):
                    onset = [1e10, 0]
                    for muscle in self.muscleMapping[m_type]:
                        dat = emg[muscle][i]
                        temp = self._locateMuscleOnset(dat,testplots)
                        if temp[0] < onset[0]:
                            onset = temp
                    epochOnsets.append(onset)
                output[m_type] = epochOnsets
                
        return output
    def _validateSaveDir(self):
        if not os.path.exists(self.saveDir):
            os.mkdir(self.saveDir)
            print(f'writing {self.saveDir} as save path')
        else:
            print('path exists')
    def _sEEG_epochPSDs(self,freqs):
        move = pd.DataFrame()
        cols = self.move_epochs.columns
        window = sig.get_window('hann',Nx=self.fs)
        f = [i for i in range(1,301)]
        f = np.array(f)
        for col in cols :
            if type(col)==int:
                move[col] = self.move_epochs[col].apply(lambda x:single_channel_pwelch(x,self.fs,window,f_bound=freqs))
            else:
                move[col] = self.move_epochs[col]
        rest = pd.DataFrame()
        cols = self.move_epochs.columns
        for col in cols :
            if type(col)==int:
                rest[col] = self.rest_epochs[col].apply(lambda x:single_channel_pwelch(x,self.fs,window,f_bound=freqs))
            else:
                rest[col] = self.rest_epochs[col]
        return move,rest,f    
    def _epoch_PSD_average(self,df):
        out = pd.DataFrame()
        movements = list(self.muscleMapping.keys())
        for m in movements:
            d = df.query('movement == @m & type == "sEEG"').copy()
            numCols = [col for col in d if type(col) == int]

            avgs = d[numCols].apply(lambda x: (np.array(np.mean(x.to_numpy(),axis=0))),axis=1)
            std = d[numCols].apply(lambda x: (np.array(np.var(x.to_numpy(),axis=0))),axis=1)
            d['avg'] = avgs
            d['std'] = std
            out = pd.concat([out,d],ignore_index=True)
        targetKeys = [i for i in out.columns if type(i)==str]
        return out[targetKeys], out
    def _epoch_PSD_std(self,move,rest):
        strCols = ['name','type','movement']
        out = pd.DataFrame()
        res = pd.DataFrame()
        res['ms'] = move['std']
        res['rs'] = rest['std']
        res = res.apply(lambda x: (np.mean(x.to_numpy())),axis=1)
        out[strCols] = move[strCols]
        out['avg_std'] = res # note this avg std is across both movement and rest trials, and is currently computed element wise across the frequency band
        return out

    def _globalPSD_normalize(self, channel_norm:bool=True):
        """
        normalize brain recroding PSDs across entire session
        ----------
        Parameters
        ----------
        channel_norm: bool, default is True
            if True, return a dict of each normalized channel
            if False, return an dict of each channel filled with the average PSD of all channels    
        """
        # # numCols = [col for col in df2 if type(col) == int]
        # # sub_df2 = df2[numCols].copy()
        # # new_numCols = [col+10 for col in sub_df2]
        # aggregate = pd.DataFrame()
        # aggregate['av1'] = df1['avg']
        # aggregate['av2'] = df2['avg']
        # av1 = aggregate['av1'].mean()
        # av2 = aggregate['av2'].mean()
        # out = np.mean([av1,av2],axis=0)
        # # aggregate[new_numCols] = sub_df2[numCols]
        data = {}
        for i in self.session_data.data['sEEG'].values():
            data.update(i)
        
        window = window = sig.get_window('hann',Nx=self.fs)
        keys = []
        psds = []
        for k,v in data.items():
            pxx = single_channel_pwelch(v,self.fs,window)
            psds.append(pxx)
            keys.append(k)
        if channel_norm:
            out = {k:v for k,v in zip(keys,psds)}    
        else:
            avg = np.average(psds,axis=0)
            out = {k:avg for k in keys}
        return out
    
    def normalizePSDs(self,data,avgs):
        out = data.copy()
        out['global'] = data.loc[:]['name'].map(avgs)
        out['normalized'] = data.loc[:]['avg'] / out.loc[:]['global']
        return out
    def sliceDataFrame(self,df,slice,key):
        df[key] = df[key].apply(lambda x: sliceArray(x,slice))
        return df
    def rsquared_analysis(self, saveMAT:bool=False,freqRange:list = [1,300],plotSection:bool=False):
        motor, rest, f = self._sEEG_epochPSDs([freqRange[0],freqRange[1]+1])
        gamma_slice = [np.where(f==65)[0][0],np.where(f==115)[0][0]+1]
        motor, fullMotor = self._epoch_PSD_average(motor)
        rest, fullRest = self._epoch_PSD_average(rest)
        g_av = self._globalPSD_normalize()
        motor = self.normalizePSDs(motor,g_av)
        rest = self.normalizePSDs(rest,g_av)
        motor_gamma = self.sliceDataFrame(motor,gamma_slice,key='normalized')
        rest_gamma = self.sliceDataFrame(rest,gamma_slice,key='normalized')
        stdev = self._epoch_PSD_std(motor,rest)
        stdev_gamma = self.sliceDataFrame(stdev,gamma_slice,key='avg_std')
        if plotSection: #leave false , but did not want to remove entirely as is useful sanity checking step 
            gamma_f = sliceArray(f,gamma_slice)
            for i in range(150,188):
                fig, (ax1, ax2,ax3) = plt.subplots(3,1)

                ax1.semilogy(f,motor_gamma.loc[i,'avg'],label="move")
                ax1.semilogy(f,rest_gamma.loc[i,'avg'] ,label="rest")
                ax3.plot(gamma_f,motor_gamma.loc[i,'normalized'],label='move')
                ax3.plot(gamma_f,rest_gamma.loc[i,'normalized'] ,label='rest')
                ax2.semilogy(f,motor_gamma.loc[i,'avg']*f,label="move")
                ax2.semilogy(f,rest_gamma.loc[i,'avg']*f ,label="rest")
                ax1.legend()
                ax2.legend()
                ax3.legend()
                fig.suptitle(f'{motor_gamma.loc[i,"name"]},{motor_gamma.loc[i,"movement"]}')
                plt.show()
        r_sq = self.compute_cross_correlations(motor_gamma,rest_gamma,stdev_gamma)
        r_pval, cohen = self.compute_power_distribution_significance(fullMotor,fullRest,gamma_slice)
        if saveMAT:
            self._validateSaveDir()
            for entry,values in r_sq.items():
                scio.savemat(self.saveDir/f'{entry}_rsq.mat',values)
        return 0
    def compute_cross_correlations(self,motor:pd.DataFrame,rest:pd.DataFrame,stdev:pd.DataFrame):
        res = {}
        channels = motor['name'].to_list()
        motor.set_index('name',inplace=True)
        rest.set_index('name',inplace=True)
        stdev.set_index('name',inplace=True)
        for i in self.muscleMapping.keys():
            m_m = motor.query('movement==@i')
            r_m = rest.query('movement==@i')
            s_m = stdev.query('movement==@i')
            temp = {}
            for chan in channels:
                m = m_m.loc[chan,'normalized']
                r = r_m.loc[chan,'normalized']
                s = s_m.loc[chan,'avg_std']
                r_sq = cross_correlation(m,r,s)
                temp[chan] = r_sq
            ref = temp.pop('REF_1_2')
            res[i] = temp
        
        return res
    def compute_power_distribution_significance(self,motor:pd.DataFrame,rest:pd.DataFrame,frequency_slice:list):
        res = {}
        d_res = {}
        channels = motor['name'].to_list()
        motor.set_index('name',inplace=True)
        rest.set_index('name',inplace=True)
        epochCols = [i for i in motor.columns if type(i)==int]
        for i in self.muscleMapping.keys():
            m_m = motor.query('movement==@i')
            r_m = rest.query('movement==@i')
            temp ={}
            d_temp = {}
            for chan in channels:
                if chan.find('REF_1_2')<0:
                    m = m_m.loc[chan,epochCols].to_numpy()
                    m_avg = epoch_powerAverage(m,frequency_slice)
                    r = r_m.loc[chan,epochCols].to_numpy()
                    r_avg = epoch_powerAverage(r,frequency_slice)
                    U,p = mannwhitneyU(m_avg,r_avg)
                    d = cohendsD(m_avg,r_avg)
                    d_temp[chan] = np.array([d,p])
                    temp[chan] = np.array([U,p])
            res[i] = temp
            d_res[i] = d_temp
        return res, d_res
    def returnSignificantChannels(self,data,dictLevels):
        for i in dictLevels:
            pass    
        return 0

def epoch_powerAverage(a,f_slice = [0]):
    averagePower = np.empty(len(a))
    for i,epoch in enumerate(a):
        if f_slice[1]:
            # print('slicing')
            averagePower[i]=np.mean(sliceArray(epoch,f_slice))
        else: 
            averagePower[i]=np.mean(epoch)
    return averagePower





def cross_correlation(m:float or np.ndarray,r:float or np.ndarray,stdev:float or np.ndarray,num_r=10,num_m=10):
    """
    
    """
    rsq = metrics.r2_score(m,r)
    N = (num_r*num_m)/((num_r+num_m)**2)
    m_in = np.mean(m)
    r_in = np.mean(r)
    variance = np.var([m,r])
    res = (m_in - r_in)**3 / (abs(m_in-r_in)*variance) * N
    return res
def single_channel_pwelch(array:np.ndarray,fs:int,window:np.ndarray,overlap=0.5,test=False,f_bound:list = [1,301]):
    """
    function to pass to a dataframe as a lambda function to perform columnwise PSDs on epoched data 
    """
    f,pxx = sig.welch(x=array,fs=fs,window=window,noverlap=int(fs*overlap),scaling='density')
    if test:
        fig,ax = method_plot(pxx,f,logy=True)
        ax.axvline(f[301])
        plt.show()
    f,pxx = f[f_bound[0]:f_bound[1]],pxx[f_bound[0]:f_bound[1]]
    return pxx

def method_plot(y,x = False, log = False, logx = False,logy=False):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    if type(x) == bool:
        ax.plot(y)
    elif logx:
        ax.semilogx(x,y)
    elif logy:
        ax.semilogy(x,y)
    elif log:
        ax.loglog(x,y)
    else:
        ax.plot(x,y)
    return fig, ax


class format_Stimulus_Presentation_Session():
    def __init__(self,loc:Path,subject,plot_stimuli=False):
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
        self.epoch_info = self.epochStimulusCode(plot_stimuli)

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
    def epochStimulusCode(self,plot_states):
        data = self.states['StimulusCode']
        moveEpochs = {}
        onset_shift = 1000
        offset_shift = 3000
        for stim in self.stimuli.values(): # get intervals for each of the stimuls codes
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
            restEpochs[k] = temp                

        # epochs['rest'] = intervals[1:]
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

def sliceArray(array, interval):
    return array[interval[0]:interval[1]]


"""Script for debugging"""


if __name__ == '__main__':
    userPath = Path(os.path.expanduser('~'))
    dataPath = userPath / "Box\Brunner Lab\DATA\SCAN_Mayo"
    subject = 'BJH041'
    session = 'post_ablation'

    a = SCAN_SingleSessionAnalysis(dataPath,subject,session,load=True,plot_stimuli=False)
    # a.export_session_EMG()
    # a.export_epochs(signalType='EMG',fname='emg')
    a.rsquared_analysis(saveMAT=False)