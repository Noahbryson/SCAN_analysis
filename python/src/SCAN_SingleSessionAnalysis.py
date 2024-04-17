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
from stimulusPresentation import format_Stimulus_Presentation_Session
from ERP_struct import ERP_struct

class SCAN_SingleSessionAnalysis(format_Stimulus_Presentation_Session):
    def __init__(self,path:str or Path,subject:str,sessionID:str,fs:int=2000,load=True,epoch_by_movement:bool=True,plot_stimuli:bool=False,gammaRange=[65,115]) -> None: # type: ignore
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
        print(f'')
        self.main_dir = path
        self.subject = subject
        self.sessionID = sessionID
        self.fs = fs

        if os.path.exists(self.main_dir/self.subject/'muscle_mapping.csv'):
            with open(self.main_dir/self.subject/'muscle_mapping.csv', 'r') as fp:
                reader = csv.reader(fp)
                self.muscleMapping = {rows[0]:rows[1:] for rows in reader}
        else:
            self.muscleMapping = {'1_Hand':['wristExtensor', 'ulnar'], '3_Foot':['TBA'],'2_Tongue':['tongue']}
        self.subjectDir = path / subject / sessionID
        dataLoc = self.subjectDir / 'preprocessed'
        self.saveRoot = self.subjectDir / 'analyzed'
        if sessionID.find('aggregate')>-1:
            HDF = True
        else:
            HDF = False
        self.gammaRange = gammaRange
        super().__init__(dataLoc,subject,plot_stimuli=plot_stimuli,HDF=HDF)
        self.epoch_info = self.epochStimulusCode_SCANtask(plot_stimuli)
        self.signalTypes = set(self.channels.values())
        self.colorPalletBest = [(62/255,108/255,179/255), (27/255,196/255,225/255), (129/255,199/255,238/255),(44/255,184/255,149/255),(0,129/255,145/255), (193/255,189/255,47/255),(200/255,200/255,200/255)]
        self.data = self._processSignals(load)
        # self.session_info.data = self.getBroadBandGamma(gammaType='wide')
        self.sessionEMG = self.data['EMG']
        self.data['sEEG'], self.ref = self.remove_references()
        self.ERP_epochs = self._epochERPs()
        self.move_epochs = self._epochData('move')
        self.rest_epochs = self._epochData('rest')
        self.motor_onset = self._EMG_activity_epochs(testplots=False)
        self.move_epochs,self.rest_epochs = self.reshape_epochs()
        if epoch_by_movement:
            self.move_epochs = self._epoch_via_EMG()
        
        print('end init')


    def _processSignals(self,load=True,bipolarSEEG=False):
        if 'timeseries_processed.pkl' in os.listdir(self.subjectDir / 'preprocessed') and load==True:
            signalGroups = readPickle(self.subjectDir / 'preprocessed' /'timeseries_processed.pkl')
            print('loaded prepocessed data')
        else:
            signalGroups = {}
            for sig in self.signalTypes:
                signalGroups[sig] = self._segmentSignals(sig)
            for sigType,data in signalGroups.items():
                if sigType == 'EMG':
                    data = self.processEMG(data)
                if sigType == 'sEEG':
                    data = self.process_sEEG(data,bipolarSEEG)
                if sigType == 'ECG':
                    data = self.processECG(data)
                if sigType == 'EEG':
                    data = self.processEEG(data)
                signalGroups[sigType] = data
            writePickle(signalGroups,self.subjectDir / 'preprocessed',fname='timeseries_processed')
            print('preprocessed the data')
        return signalGroups
    def remove_references(self):
        data = {k:v for k,v in self.data['sEEG'].items() if k.find('REF')<0}
        ref = {k:v for k,v in self.data['sEEG'].items() if k.find('REF')>=0}
        return data, ref

    def probeSignalQuality(self,channel:str=''):
        allChans = list(self.data['sEEG'].keys())
        if channel == '':
            shank = allChans[0]
            data = list(self.data['sEEG'][shank].keys())
            channel = data[0]
        else:
            shank = channel.split('_')[0]
        data = self.data['sEEG'][shank][channel]
        f,pxx = sig.welch(x=data,fs=self.fs,window=sig.get_window('hann',Nx=self.fs),scaling='density')
        t = np.linspace(0,len(data)/self.fs,len(data))
        alpha = bandpass(data,self.fs,[8,13],4)
        beta = bandpass(data,self.fs,[14,28],4)
        high_gamma = bandpass(data,self.fs,self.gammaRange,4)
        fig, (ax1,ax3,ax4,ax5) = plt.subplots(4,1,sharex=True)
        ax1.plot(t,data)
        ax1.set_title(channel)
        # ax2.semilogy(f,pxx)
        # ax2.set_ylabel('(V**2/Hz)')
        # ax2.set_xlabel('Hz')
        # ax2.set_title(f'{channel} PSD')
        # ax2.set_xlim([1,300])
        ax3.plot(t,alpha)
        ax3.set_title('alpha')
        ax4.plot(t,beta)
        ax4.set_title('beta')
        ax5.plot(t,high_gamma)
        ax5.set_title('broadband gamma')
        plt.show()

    


    def getBroadBandGamma(self,gammaType:str=''):
        data = self.data.copy()
        if gammaType.lower() =='wide':
            bandSplit = ([70,80],[80,90],[90,100],[100,110],[110,120],[120,130],[130,140],[140,150],[150,160],[160,170])
        else:
            bandSplit = ([65,75],[75,85],[85,95],[95,105],[105,115])
        
        output = {}
        for traj,chans in data['sEEG'].items():
            traj_data = {}
            for channel,stream in chans.items():
                bands = np.empty([len(bandSplit),len(stream)])
                for i,band in enumerate(bandSplit):
                    p = bandpass(stream,self.fs,Wn=band,order=3)
                    pxx = hilbert_env(p) **2
                    bands[i] = pxx
                
                stream = sum(bands)
                # ax = plt.subplot(6,1,1)
                # ax.plot(temp)
                stream = np.log10(stream)
                # ax = plt.subplot(6,1,2)
                # ax.plot(temp)
                stream = zscore_normalize(stream)
                # ax = plt.subplot(6,1,3)
                # ax.plot(temp)
                stream = moving_average_np(stream,1000)
                # temp = savitzky_golay(temp,window_size=999,order=0)
                
                # ax = plt.subplot(6,1,4)
                # ax.plot(temp)
                stream = math.e**stream
                # ax = plt.subplot(6,1,5)
                # ax.plot(temp)
                stream = stream - 1
                # ax = plt.subplot(6,1,6)
                # ax.plot(temp)
                traj_data[channel] = stream 
            output[traj] = traj_data
        output = alphaSortDict(output)
        data['sEEG'] = output
        return data
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
    def process_sEEG(self,sEEG:dict,bipolar:bool=True):
        trajectories = [key[0:2] for key in sEEG.keys()]
        trajectories = set(trajectories)
        trajectories.remove('RE')
        trajectories.add('REF')
        
        output = {}
        for traj in trajectories:
            if bipolar:
                data = [v for k,v in sEEG.items() if k.find(traj)>-1]
                traj_data = {}
                for idx,vals in enumerate(data[0:-1]):
                    label = f'{traj}_{idx+1}_{idx+2}'
                    temp = self._bipolarReference(data[idx+1],vals)
                    # temp = notch(temp,self.fs,60,30,1)
                    traj_data[label] = temp 
                output[traj] = traj_data
            else:
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
    
    def _epochERPs(self):
        """_epochERPs _summary_

        Returns:
            _type_: _description_
        """
        epochInfo = self.epoch_info
        epochs = {}
        for i,j in zip(epochInfo[0],epochInfo[1]):
            m,r = epochInfo[0][i],epochInfo[1][j]
            epochs[i] = [[p[0],p[1],q[1]] for p,q in zip(m,r)]
        epochs['info'] = ['motor onset', 'motor offset', 'rest offset']
        return epochs
    
    def extractAllERPs(self):
        """extractAllERPs _summary_

        Returns:
            _type_: _description_
        """
        epochs = {}
        for k,v in self.ERP_epochs.items(): # epoch information (muscle, list of intervals)
            if k.find('info') < 0:
                signals = {}
                for sigType, values in self.data.items(): # (type of signal, dictionary of all data)
                    trajectories = {}
                    for traj, chan in values.items(): # (name of specific trajectory, recording sites on the trajectory)
                        channel = {loc: [[data[on:end+1],[0,restOn-on+1,end-on+1]] for (on,restOn, end) in v] for loc, data in chan.items()} # dict comprehension to build epochs from intervals on each channel on a trajectory
                        if sigType in signals:
                            signals[sigType].update(channel) 
                        else:
                            signals[sigType] = channel
                epochs[k] = signals
        gammaERP = self.highGamma_ERP(epochs)
        gammaERP.plotAverages_per_trajectory(self.subject)
        plt.show()
        return epochs
                
    def highGamma_ERP(self,data,power: bool=True)-> ERP_struct:
        inputData = data
        output = self.general_ERP(inputData, self.gammaRange,power=power)
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
        aggregateData = {}
        for mv, data in epochs.items():
            averages = {}
            raw = {}
            stdevs = {}
            agg = data['sEEG']
            agg.update(data['EMG'])
            gen = ([channel,vals] for channel,vals in agg.items() if channel.find('REF')<0)
            for i in gen:
                channel = i[0]
                temp = []
                for epoch,info in i[1]:
                    if not any(char.isdigit() for char in channel):
                        "Parsing for EMG channels"
                        if channel[0:3] != 'EMG': 
                            channel = f'EMG_{channel}'       
                    elif power:
                        "Getting Timeseries Power"
                        # if filterBand[1] - filterBand[0] > 10:
                        #     filtBands = [[filterBand[0]+10*i,filterBand[0]+10*(i+1)] for i in range(int((filterBand[1]-filterBand[0])/10)-1)]
                        # else:
                        #     filtBands = [filterBand]
                        # hold = []
                        # for b in filtBands:
                        #     d = bandpass(epoch,self.fs,b,order=4)
                        #     hold.append(hilbert_env(d)**2)
                        # epoch = sum(hold)
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
        
    def _segmentSignals(self,sigType:dict):
        out = {}
        for k,i in self.channels.items():
            if i == sigType:
                out[k] = self.data[k]
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
            dir = self.saveRoot
            scio.savemat(dir/f'{fname}_{i}.mat',out)
        return 0
    def export_session_EMG(self):
        dat = self.sessionEMG['EMG']
        scio.savemat(self.saveRoot/'fullEMG.mat',dat)

    def reshape_epochs(self):
        move = pd.DataFrame()
        for k,d in self.move_epochs.items():
            move = pd.concat([move,self.epochs_to_df(d,k)])
        typeCol = ['move' for _ in range(move.shape[0])]
        move['class'] = typeCol
        rest = pd.DataFrame()
        for k,d in self.rest_epochs.items():
            rest = pd.concat([rest,self.epochs_to_df(d,k)])
        typeCol = ['rest' for _ in range(rest.shape[0])]
        rest['class'] = typeCol
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
        for muscle, intervals in self.epoch_info[idx].items(): # epoch information (muscle, list of intervals)
            signals = {}
            for sigType, values in self.data.items(): # (type of signal, dictionary of all data)
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
    def _validateDir(self,subDir=''):
        if subDir != '':
            saveDir = self.saveRoot/subDir
        else:
            saveDir=self.saveRoot
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
            print(f'writing {saveDir} as save path')
        else:
            print('path exists')
        return saveDir
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
        for i in self.data['sEEG'].values():
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
    def taskPowerCorrelation_analysis(self, saveMAT:bool=False,freqRange:list = [1,300],plotSection:bool=False):
        motor, rest, f = self._sEEG_epochPSDs([freqRange[0],freqRange[1]+1])
        gamma_slice = [np.where(f==self.gammaRange[0])[0][0],np.where(f==self.gammaRange[-1])[0][0]+1]
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
        p, U_res, d_res,roc_res = self.compute_power_distribution_significance(fullMotor,fullRest,gamma_slice)
        if saveMAT:
            saveDir = self._validateDir()
            for entry,values in r_sq.items():
                scio.savemat(saveDir/f'{entry}_rsq.mat',values)
        return r_sq, p, U_res, d_res,roc_res
    
    
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
                r_sq = signed_cross_correlation(m,r,s)
                temp[chan] = r_sq
            out = {i:temp[i] for i in temp.keys() if i.find('REF')<0}
            res[i] = out
        
        return res
    def compute_power_distribution_significance(self,motor:pd.DataFrame,rest:pd.DataFrame,frequency_slice:list):
        U_res = {}
        d_res = {}
        roc_res = {}
        p_res = {}
        channels = motor['name'].to_list()
        motor.set_index('name',inplace=True)
        rest.set_index('name',inplace=True)
        epochCols = [i for i in motor.columns if type(i)==int]
        for i in self.muscleMapping.keys():
            m_m = motor.query('movement==@i')
            r_m = rest.query('movement==@i')
            temp ={}
            d_temp = {}
            roc_temp = {}
            p_temp = {}
            for chan in channels:
                if chan.find('REF_1_2')<0:
                    m = m_m.loc[chan,epochCols].to_numpy()
                    m_avg = epoch_powerAverage(m,frequency_slice)
                    r = r_m.loc[chan,epochCols].to_numpy()
                    r_avg = epoch_powerAverage(r,frequency_slice)
                    U,p = mannwhitneyU(m_avg,r_avg)
                    d = cohendsD(m_avg,r_avg)
                    roc = calc_ROC(m_avg,r_avg,plot=False)
                    d_temp[chan] = d
                    temp[chan] = U
                    roc_temp[chan] = roc
                    p_temp[chan] = p
            U_res[i] = temp
            d_res[i] = d_temp
            roc_res[i] = roc_temp
            p_res[i] = p_temp
        return p_res, U_res, d_res,roc_res
    def returnSignificantChannels_perTask(self,data,alpha=0.05,saveTXT=False):
        res = {}
        for cond, items in data.items():
            temp = []
            for c, p in items.items():
                if p < alpha:
                    temp.append(c)
            res[cond] = temp
        targetTrajs = 'IJKLM'
        if saveTXT:
            saveDir = self._validateDir(subDir='metrics')
            with open(saveDir/'sig_chans.txt', 'w') as fp:
                for cond in res:
                    count = 0
                    fp.write(f'\n{cond}\n')
                    for i in res[cond]:
                        fp.write(f'{i},')
                        if targetTrajs.find(i[0].upper()) >-1:
                            count +=1
                    print(f'{cond} num target chans: {count}')
        return res
    
    def returnSignificantLocations(self,pvalDict,alpha):
        data = self.reshapeEffect(pvalDict)
        sig = []
        no_sig = []
        for channel, vals in data.items():
            ps = list(vals.values())
            ps = np.array(ps)
            if np.any(ps < alpha):
                sig.append(channel)
            else:
                no_sig.append(channel)
        return sig, no_sig
            

    def aggregateResults(self,r_sq, p, U_res, d_res,roc_res,saveMAT=False):
        sigDict = self.returnSignificantChannels_perTask(p,alpha = 0.05)
        sig_r = {}
        sig_d = {}
        sig_roc = {}
        sig_U = {}
        for cond, chans in sigDict.items():
            sig_r   [cond]= parseDictViaKeys(   r_sq[cond], keys=chans)
            sig_d   [cond]= parseDictViaKeys(  d_res[cond], keys=chans)
            sig_roc [cond]= parseDictViaKeys(roc_res[cond], keys=chans)
            sig_U   [cond]= parseDictViaKeys(  U_res[cond], keys=chans)
        outs = [sig_r,sig_d,sig_roc,sig_U,p]
        lab = ['rsq','cohen','roc','U','pval']
        if saveMAT:
            for i,l in zip(outs,lab):
                self.dict2mat(i,name=f'{l}_sig')
        return sig_r,sig_d,sig_roc,sig_U
    
    def visualizeMetrics(self,sig_r,sig_d,sig_roc,sig_U,numBins=10):
        labels = ['task','chan','metric','value']
        temp = []
        out = pd.DataFrame()
        tasks = list(sig_r.keys())
        for i,task in enumerate(tasks):
            
            for k in sig_r[task].keys():
                cohenNorm = max(list(sig_d[task].values()))
                temp.append([task,k,'R_squared',(sig_r[task][k]+1)/2])
                temp.append([task,k,"Cohen's d",((sig_d[task][k]/cohenNorm)+1)/2])
                # temp.append([task,k,'rsq',sig_r[task][k]])
                # temp.append([task,k,'cohen',sig_d[task][k]])
                temp.append([task,k,'AUC',sig_roc[task][k]])
                # temp.append([task,k,'U',sig_U[task][k]/100])
            # fig = plt.figure(num=f'{self.session} {task}_count')
            # ax = plt.gca()
            df = pd.DataFrame(temp,columns=labels)
            # df.sort_values('value',inplace=True)
            df.sort_values('metric',inplace=True)
            # sns.stripplot(df,x='metric',y='value',ax=ax)
            # ax.set_title('Metrics')
            # ax.tick_params(labelrotation=90)

            fig = plt.figure(num=f'{self.sessionID} {task}_hist')
            ax = plt.gca()
            lab = task.split('_')
            lab2 = self.sessionID.split('_')
            sns.histplot(df,x='value',hue='metric',ax=ax,stat='percent',kde=True,bins=numBins)
            ax.set_title(f'{lab[-1]}, {lab2} ')
            ax.axvline(0.5,label='Increase Boundary',c=(0,0,0),linestyle='--')
            out = pd.concat([out,df],ignore_index=True)
        # for a in ax.flat:
        #     a.label_outer()
        plt.show()
        return out


    def dict2mat(self,dic,name='', saveFolder='significance'):
        saveDir = self._validateDir(saveFolder)
        for entry,values in dic.items():
            scio.savemat(saveDir/f'{name}_{entry}.mat',values)

    def runEffectClusters(self, effectDict, thresh, dataSubset:list=[],title='',for_subplot:bool=False, ax: bool or plt.axes=False,  # type: ignore
                          clusterFlag:bool= False, num_clusters=5,exportFlag:bool=False,effectLabel:str=''):
        """plot effect sizes for the 3 movements as 3D scatter plot. 
            x-axis is hand
            y-axis is foot
            z-axis is tongue
            ---------
            Parameters
            ---------
            effectDict: dictionary
                output dictionary of one variable from taskPowerCorrelation_analysis
            thresh: float
                minimum value of effect size required for each condition to be flagged as an intereffector
                not this value is a manual test and not an effective substitute for clustering
            dataSubset: list
                list of keys to parse the effectDict dictionary by. keys in this list are included in the visualization, keys not in the list are excluded.
            title: str
                title string of the axes created by this function.
            for_subplot: bool
                wheter or not this function is being displayed in a different subplot.
            ax : bool or plt.axes
                axes passed if for_subplot==True, else false
            cluster: bool
                flag to perform knn clustering on the data    
            """
        data = self.reshapeEffect(effectDict)
        if len(dataSubset) > 0:
            print('parsing via keys')
            data = parseDictViaKeys(data=data,keys=dataSubset)
        n_samp = len(data)
        print(f'{title} num_chans:{n_samp}')
        xyz = np.empty([n_samp,3])
        if not for_subplot:
            fig = plt.figure(f'{title} effect projection')
            ax = fig.add_subplot(projection='3d')
        chan = ''
        c = (0,0,1)
        cflag = False
        legend = []
        threshLabs = [] 
        if clusterFlag:
            cluster_res = self.kmeans_cluster(data,num_clusters)
            cluster_labs = cluster_res.labels_
            cluster_dist = np.zeros(cluster_res.n_clusters)
            for i,dist in enumerate(cluster_res.cluster_centers_):
                cluster_dist[i] = euclidean_distance(dist)
            max_res = [euclidean_distance(i) for i in cluster_res.cluster_centers_ if min(i) > thresh] # need this step to remove clusters which center about a decrease in power 
            SCAN_cluster = np.where(cluster_dist==max(max_res))
            SCAN_euclid = cluster_dist[SCAN_cluster]
            print(f'\n\nSCAN Cluster is {SCAN_cluster},\ncoordinates: {cluster_res.cluster_centers_[SCAN_cluster]}\n distance: {SCAN_euclid}')
            cmap = [self.colorPalletBest[i] for i in range(len(np.unique(cluster_labs)))]
        for i,(k,v) in enumerate(data.items()):
            # locs = v.keys()
            h = v['Hand'] # x -> hand   
            f = v['Foot'] # y -> foot
            t = v['Tongue'] # z -> tongue
            xyz[i] = np.array([h,f,t])
            euclid = euclidean_distance(xyz[i])
            if chan != k[0:2]:
                chan = k[0:2]
                if cflag:
                    c = (i/n_samp,0,1)
                    cflag = False
                else:
                    c = (.4,i/n_samp,0)
                    cflag = True
                
                legend.append(chan)
                lab = chan
            else:
                lab = '_'
            if clusterFlag:
                c_lab = cluster_labs[i]
                c = cmap[c_lab]
                if cluster_labs[i] == SCAN_cluster:
                    print(f'{k}: {euclid} dist\nhand {round(h,2)} foot {round(f,2)} tongue {round(t,2)}')
                    threshLabs.append([k,i])
                    print(f'cluster {cluster_labs[i]}')
                    size = 50
                else:
                    size = 10
            elif(clusterFlag == False and h > thresh and f > thresh and t > thresh):
                
                print(f'{k}: {euclid} dist\nhand {round(h,2)} foot {round(f,2)} tongue {round(t,2)}')
                threshLabs.append([k,i])
                size = 50
            else: 
                size = 10
            
            ax.scatter(xs=h,ys=f,zs=t,s = size,color=c,label=lab)
        threshDict = {f'{k[0].split("_")[0]}{k[0].split("_")[1]}':xyz[k[1]] for k in threshLabs}
        if clusterFlag and exportFlag:
            filename = f'{self.subject}_{self.sessionID}_{title}_{effectLabel}_cluster_result.mat'
            self._validateDir(self.saveRoot)
            output = {'clusterRes':threshDict}
            scio.savemat(self.saveRoot/filename,mdict=output,format='5')
        # ax.scatter(xs=xyz[:,0],ys=xyz[:,1],zs=xyz[:,2])
        axLine = np.linspace(-thresh*10,thresh*10,10)
        offAx = np.linspace(0,0,10)
        ax.plot(xs=axLine,ys=offAx, zs=offAx, c=(0,0,0))
        ax.plot(xs=offAx, ys=axLine,zs=offAx, c=(0,0,0))
        ax.plot(xs=offAx, ys=offAx, zs=axLine, c=(0,0,0))
        ax.set_xlabel(f'hand {effectLabel}')
        ax.set_ylabel(f'foot {effectLabel}')
        ax.set_zlabel(f'tongue {effectLabel}')
        ax.set_title(f'{self.subject}\n{title} data cluster')
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        # ax.set_zlim(0,1)
        # ax.legend()
        ax.grid(False)
        # plt.grid(visible=False)
        # sns.despine()
        return ax,threshDict
    
    def kmeans_cluster(self,data:dict,num_clusters:int)->KMeans:
        channels = data.keys()
        n_samp = len(data)
        xyz = np.empty([n_samp,3])
        for i,(k,v) in enumerate(data.items()):
            # locs = v.keys()
            h = v['Hand'] # x -> hand
            f = v['Foot'] # y -> foot
            t = v['Tongue'] # z -> tongue
            xyz[i] = np.array([h,f,t])


        cluster = KMeans(n_clusters=num_clusters,random_state=0, n_init='auto')
        cluster.fit(xyz)
        labels = cluster.labels_
        cluters_accessed = len(np.unique(labels))        
        return cluster


    def plotAllEffects(self,threshholds:list,):
        """
        ----------
        Parameters
        ----------
            effectSizeDict: list(dict)
                list of dictionaries containing the different effects for each channel and each test run. 
            threshholds: list
                list of minimum values of effect size required, paired to effect size types, for each condition to be flagged as an intereffector.
                index of this must be equal to index of effectSizeDict passed
                not this value is a manual test and not an effective substitute for clustering
            dataSubset: list
                list of keys to parse the effectDict dictionary by. keys in this list are included in the visualization, keys not in the list are excluded.
            title: str
                title string of the axes created by this function."""
        pass
        
    def reshapeEffect(self,effect):
        out = {}
        for k,v in effect.items():
            for c, r in v.items():
                temp = {k.split('_')[-1]:r}
                if c in out.keys():
                    out[c].update(temp)
                else:
                    out[c] = temp
        return out

def epoch_powerAverage(a,f_slice = [0]):
    averagePower = np.empty(len(a))
    for i,epoch in enumerate(a):
        if f_slice[1]:
            # print('slicing')
            averagePower[i]=np.mean(sliceArray(epoch,f_slice))
        else: 
            averagePower[i]=np.mean(epoch)
    return averagePower
def parseDictViaKeys(data,keys):
    res = {k:data[k] for k in keys}
    return res
def signed_cross_correlation(m:float or np.ndarray,r:float or np.ndarray,stdev:float or np.ndarray,num_r=10,num_m=10): # type: ignore
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
        x = np.linspace(0,1,len(y))
        xlab = 'au'
    else:
        xlab = 'actual scale'
    if logx:
        ax.semilogx(x,y)
        ax.set_xlabel(xlab)
    elif logy:
        ax.semilogy(x,y)
        ax.set_xlabel(xlab)
    elif log:
        ax.loglog(x,y)
        ax.set_xlabel(xlab)
    else:
        ax.plot(x,y)
        ax.set_xlabel(xlab)
    return fig, ax

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

def writePickle(struct,fpath:Path,fname):
    fname = fpath / f'{fname}.pkl'
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
    import platform
    localEnv = platform.system()
    userPath = Path(os.path.expanduser('~'))
    if localEnv == 'Windows':
        dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
    else:
        dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"
    subject = 'BJH045'
    session = 'pre_ablation'
    # session = 'post_ablation'
    session = 'aggregate'
    gammaRange = [70,170]
    a = SCAN_SingleSessionAnalysis(dataPath,subject,session,load=True,plot_stimuli=False,gammaRange=gammaRange)
    a.extractAllERPs()
    # a.probeSignalQuality('OR_7_8')
    # a.export_session_EMG()
    # a.export_epochs(signalType='EMG',fname='emg')
    r_sq, p_vals, U_res, d_res,roc_res = a.taskPowerCorrelation_analysis(saveMAT=False)
    fig = plt.figure()
    row = 2; col =1
    ax = fig.add_subplot(row, col, 1, projection='3d')
    ax.view_init(elev=30, azim=105, roll=0)
    a.runEffectClusters(r_sq,0.2,title='all channels',clusterFlag=True,ax=ax,for_subplot=True, effectLabel='(d)')
    sig_chans, nonsig_chans = a.returnSignificantLocations(p_vals,alpha=0.05)
    ax2 = fig.add_subplot(row, col, 2, projection='3d')
    ax2.view_init(elev=30, azim=105, roll=0)
    a.runEffectClusters(r_sq,0.2,dataSubset=sig_chans,title='significant channels',clusterFlag=True, num_clusters=5,ax=ax2,for_subplot=True,effectLabel='(d)',exportFlag=True)
    # plt.show()
    sig_r,sig_d,sig_roc,sig_U = a.aggregateResults(r_sq, p_vals, U_res, d_res,roc_res,saveMAT=False)
    # a.scatterMetrics(sig_r,sig_d,sig_roc,sig_U) # significant Channels
    a.visualizeMetrics(r_sq, d_res,roc_res,U_res,numBins=40) # all channels