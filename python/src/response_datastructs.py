import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from sklearn import decomposition, metrics, preprocessing
import pandas as pd
from functions.stat_methods import cohendsD, mannwhitneyU, signed_cross_correlation
from functions.filters import sliceArray
import math
import distinctipy
import random
from typing import List


class ERP_struct():
      def __init__(self,data,filterband,power,fs):
            self.filterband = filterband
            self.fs=fs
            if power:
                  self.method = 'power'
                  self.unit = 'Z-scored power'
            else:
                  self.method = 'voltage'
                  self.unit = 'uV'
            datK = list(data.keys()) # get keys for dynamic grabbing of channel names.
            self.cs = distinctipy.get_colors(len(datK),pastel_factor=0.5,rng=random.seed(35)) #generate random colors with same seed.
            self.trajectories = set([i[0:2] for i in data[datK[0]][0].keys()])
            channels = [i for i in data[datK[0]][0].keys()]
            self.mapping = {}
            for i in self.trajectories:
                  self.mapping[i] = [j for j in channels if j.find(i)>=0]
            self.num_traj = len(self.trajectories)
            self.data,self.datalabels= self._reshape_data(data,channels)            
      
      def _reshape_data(self,data: dict,channels):
            output = {}
            labels = ['avg','stdev','raw']
            for c in channels:
                  temp = {}
                  for k,v in data.items():
                        avg = v[0][c]
                        stdev = v[1][c]
                        raw = v[2][c]
                        # CI = confidence_interval(a=raw,alpha=0.95)
                        temp[k] = [avg,stdev,raw]
                  output[c] = temp
            return output, labels
      
      
      def plotAverages_per_trajectory(self,subject,timeLag:float=0.5)->List[plt.Figure]:
            figs = []
            for k,v in self.mapping.items():
                  numChan = len(v)
                  nCol = 4
                  fullArr = math.ceil(numChan/nCol) * nCol
                  plotShape = np.shape(np.zeros(fullArr).reshape((-1,nCol)))
                  
                  data = [self.data[x] for x in v]
                  title = f'{subject} {k}, {self.method}, {self.filterband} Hz ERPs'
                  fig, axs = plt.subplots(plotShape[0],plotShape[1],num = title,sharex=True,sharey=True)
                  for i,(vals,ax) in enumerate(zip(data,fig.axes)):
                        keys = list(vals.keys())
                        
                        t = np.linspace(-timeLag,len(vals[keys[0]][0])/self.fs-timeLag,len(vals[keys[0]][0]))                        
                        # a.set_visible(True)
                        for j,k in enumerate(keys):
                              ax.plot(t,vals[k][0], label=k,color=self.cs[j])
                              plot_range_on_curve(t,vals[k][0],vals[keys[2]][1],ax,color=self.cs[j])
                        ax.legend()
                        ax.set_title(v[i])
                        if k != 'EM':
                              ax.set_ylabel(f'{self.unit}')
                        
                  for a in axs.flat:
                        # a.label_outer()
                        if not a.has_data():
                              fig.delaxes(a)
                        else:
                              a.set_xlabel('time (s)')
                  figs.append(fig)
            return figs
def plot_range_on_curve(t,curve, bounds, ax:plt.axes,color):
      upper = np.add(curve,bounds)
      lower = np.subtract(curve,bounds)
      # x = np.linspace(0,len(upper),len(upper))
      ax.fill_between(t,upper,lower,color=color,alpha=0.2,label='_')
      return ax


class spectrumResponses():
      def __init__(self,data:dict, fs:int, normalizingDict:dict, filterband:list = [70,170]) -> None:
            self.filterband = filterband
            self.data = data
            self.fs = fs
            self.windowProportion = 0.5#multiply by data len during pwelch for proportion of data.
            taskPSDs = self._computeSpectrograms(normalDict=normalizingDict)
            tasks = [i for i in taskPSDs.keys()]
            self.psdDf = self._makeDataFrame(taskPSDs)
            self.taskRes = self.getTaskResults(self.psdDf,tasks)


      def _makeDataFrame(self,taskPSDs:dict):
            columns = ['taskType','TaskActive','Channel','f','PSD','normPSD','trial']
            res = []
            for k,v in taskPSDs.items():
                  taskData = v['task']
                  restData = v['rest']
                  for t,r in zip(v['task'],v['rest']):
                              for trial,(f,p,pn) in enumerate(zip(taskData[t][0],taskData[t][1],taskData[t][2])):
                                    res.append([k,True,t,f,p,pn,trial+1])
                              for trial,(f,p,pn) in enumerate(zip(restData[r][0],restData[r][1],restData[r][2])):
                                    res.append([k,False,r,f,p,pn,trial+1])
            df = pd.DataFrame(res,columns=columns)
            df['gamma'] =      df.apply(lambda x: np.mean(sliceArray(x['PSD'],self.filterband)),axis=1)
            df['norm_gamma'] = df.apply(lambda x: np.mean(sliceArray(x['normPSD'],self.filterband)),axis=1)
            return df

      def getTaskResults(self,taskDf:pd.DataFrame, tasks):
            channels = set(taskDf['Channel'].to_list())
            output = []
            columns = ['channel','task','rsq','d','U','p']
            for c in channels:
                  for task in tasks:
                        task_on  = taskDf.loc[(taskDf['taskType']==task) & (taskDf['TaskActive']==True) & (taskDf['Channel']==c)]
                        task_off  = taskDf.loc[(taskDf['taskType']==task) & (taskDf['TaskActive']==False) & (taskDf['Channel']==c)]
                        m=task_on['norm_gamma'].to_numpy()
                        r=task_off['norm_gamma'].to_numpy()
                        r_sq = signed_cross_correlation(m=m,r=r,num_r=len(r),num_m=len(m)) 
                        d = cohendsD(m,r)
                        res = mannwhitneyU(m,r)
                        output.append([c,task,r_sq,d,res.statistic,res.pvalue])
            df=pd.DataFrame(output,columns=columns)
            return df

      def getSignificantChannels(self):
            # channels = self.taskRes.loc[self.taskRes['p']<0.05]['channel'].to_list()
            channels = self.taskRes['channel'].to_list()
            channels = set(channels)
            temp = self.taskRes.loc[self.taskRes['channel'].isin(channels)]
            out_r, out_d = {},{}
            for c in channels:
                  data = temp.loc[temp['channel']==c]
                  data_r = {k:v for k,v in zip(data['task'],data['rsq'].to_numpy())}
                  data_d = {k:v for k,v in zip(data['task'],data['rsq'].to_numpy())}
                  out_d[c] =data_d
                  out_r[c] =data_r
            dfrsq = pd.DataFrame.from_dict(out_r,orient='index')
            dfd = pd.DataFrame.from_dict(out_d,orient='index')
            return dfd, dfrsq

      def clusterChannels(self):
            dfd, dfrsq = self.getSignificantChannels()
            dfd[dfd.columns] = preprocessing.StandardScaler().fit_transform(dfd)
            n_comps = 3
            pca = decomposition.PCA(n_components=n_comps)
            pca_res = pca.fit_transform(dfd)
            pcaDf = pd.DataFrame(abs(pca.components_),columns=dfd.columns,index=[f'PC{i+1}' for i in range(n_comps)])
            ax = plt.figure().add_subplot(1,1,1,projection='3d')
            for r in pca_res:
                  ax.scatter(r[0],r[1],color=(0,0,0),alpha=0.4)
            plt.show()
            return 0




      def _computeSpectrograms(self,normalDict):
            output = {}
            for k,v in self.data.items():
                  task = v['task']['sEEG']
                  rest = v['rest']['sEEG']
                  task_res = {}
                  rest_res = {}
                  for t,r in zip(task,rest):
                        t_temp, r_temp = [], []
                        t_norm, r_norm = [], []
                        t_f, r_f = [], []
                        for t_int,r_int in zip(task[t],rest[r]):
                              # window = sig.get_window('hann',Nx= int(self.windowProportion*len(t_int)))
                              window = sig.get_window('hann',Nx=self.fs)
                              f,Pxx = sig.welch(t_int,fs=self.fs,window=window)
                              t_temp.append(Pxx)
                              t_norm.append(Pxx / normalDict[t][-1])
                              t_f.append(f)
                              
                              # window = sig.get_window('hann',Nx= int(self.windowProportion*len(r_int)))
                              window = sig.get_window('hann',Nx= int(self.fs))
                              f,Pxx = sig.welch(r_int,fs=self.fs,window=window)
                              r_temp.append(Pxx)
                              r_norm.append(Pxx / normalDict[t][-1])
                              r_f.append(f)
                        task_res[t] = [t_f,t_temp,t_norm]
                        rest_res[t] = [r_f,r_temp,r_norm]
                        
                  output[k] = {'task':task_res,'rest':rest_res}
            # x = plt.figure()
            # x.suptitle('IL_3')
            # ax = x.add_subplot(111)
            # ax.semilogx(output['pinky']['task']['IL_3'][0][0],output['pinky']['task']['IL_3'][2][0])
            # ax.semilogx(output['pinky']['rest']['IL_3'][0][0],output['pinky']['rest']['IL_3'][2][0],c=(0,1,0))                             
            # plt.close()
            # breakpoint()
            return output
      
      def getFFTWindow(self):
            print('returned window is the proportion of the datalength')
            return self.windowProportion
      def setFFTWindow(self,windowProportion):
            self.windowProportion = windowProportion