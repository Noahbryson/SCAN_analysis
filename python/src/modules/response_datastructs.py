import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from sklearn import decomposition, metrics, cluster,preprocessing
import pandas as pd
from functions.stat_methods import cohendsD, mannwhitneyU, signed_cross_correlation
from functions.filters import sliceArray
import math
import distinctipy
import random
from typing import List
import pickle
from pathlib import Path


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
            datK = list(data.keys()) # get keys for dynamic grabbing of task names.
            self.cs = distinctipy.get_colors(len(datK),pastel_factor=0.5,rng=random.seed(35)) #generate random colors with same seed.
            self.trajectories = set([i[0:2] for i in data[datK[0]][0].keys()])
            channels = [i for i in data[datK[0]][0].keys()]
            self.mapping = {}
            for i in self.trajectories:
                  temp = [j for j in channels if j.find(i)>=0]
                  if i == 'EM':
                        temp.sort()
                  self.mapping[i] = temp
            self.num_traj = len(self.trajectories)
            self.data, self.ERP_ANOVA, self.datalabels= self._reshape_data(data,channels)            
      
      def _reshape_data(self,data: dict,channels):
            output = {}
            outstat = {}
            labels = ['avg','stdev','raw']
            for c in channels:
                  temp = {}
                  for k,v in data.items():
                        avg = v[0][c]
                        stdev = v[1][c]
                        raw = v[2][c]
                        # CI = confidence_interval(a=raw,alpha=0.95)
                        temp[k] = [avg,stdev,raw]
                        self.samplen = len(avg)
                  stat_res = self.ERP_comparison(temp,self.samplen)
                  outstat[c] = stat_res
                  output[c] = temp
            return output, outstat, labels
      
      def ERP_comparison(self,data: dict,sampLen: int):
            vals = []
            bin_length = 500
            bin_samps = int(bin_length * self.fs / 1000)
            num_bins = int(math.floor(sampLen / bin_samps))
            for v in data.values():
                  b = binned_ERP(v[-1],num_bins=num_bins)
                  vals.append(b)
            res = st.f_oneway(vals[0],vals[1],vals[2],axis=0)
                  
            
            
            return res
      
      
      def plotAverages_per_trajectory(self,subject,timeLag:float=1)->List[plt.Figure]:
            figs = []
            for k,v in self.mapping.items():
                  numChan = len(v)
                  nCol = 4
                  fullArr = math.ceil(numChan/nCol) * nCol
                  plotShape = np.shape(np.zeros(fullArr).reshape((-1,nCol)))
                  
                  data = [self.data[x] for x in v]
                  stats = [self.ERP_ANOVA[x] for x in v]
                  stattime = np.linspace(0.5-timeLag,self.samplen/self.fs - timeLag,12)
                  title = f'{subject} {k}, {self.method}, {self.filterband} Hz ERPs'
                  figname = f'{subject}_{k}_{self.method}_{self.filterband}_Hz_ERPs'
                  fig, axs = plt.subplots(plotShape[0],plotShape[1],num =figname,sharex=True,sharey=True,figsize=(20,15))
                  for i,(vals,ax) in enumerate(zip(data,fig.axes)):
                        keys = list(vals.keys())
                        statPlot = np.array([[a,-1] for a,y in zip(stattime,stats[i].pvalue) if y <=0.05]).reshape(-1,2).T
                        
                        t = np.linspace(-timeLag,len(vals[keys[0]][0])/self.fs-timeLag,len(vals[keys[0]][0]))                        
                        # a.set_visible(True)
                        for j,p in enumerate(keys):
                              ax.plot(t,vals[p][0], label=p,color=self.cs[j])
                              plot_range_on_curve(t,vals[p][0],vals[keys[2]][1],ax,color=self.cs[j])
                        ax.scatter(statPlot[0],statPlot[1], color=(0,0,0))
                        ax.legend()
                        ax.set_title(v[i])
                        # ax.set_ylim(-0.5,2)
                        if k != 'EM':
                              ax.set_ylabel(f'{self.unit}')
                              ax.set_ylim(-1.1,2)
                        
                        ax.axvline(0,-1,2,c=(0,0,0),alpha = 0.5)
                        
                  for a in axs.flat:
                        # a.label_outer()
                        if not a.has_data():
                              fig.delaxes(a)
                        else:
                              a.set_xlabel('time (s)')
                  figs.append(figname)
            return figs
      
      def emg_isolation(self,subject,task_mapping,timeLag:float=1,save:bool=False, plot: bool=False,figPath = '', dataPath =''):
            if not 'EM' in self.mapping:
                  print('no EMG present')
                  return False
            if dataPath != '' and isinstance(dataPath,str):
                  dataPath = Path
            EMG = {x.split('_')[-1]:self.data[x] for x in self.mapping['EM']}
            k1 = list(EMG.keys()); k2 = list(EMG[k1[0]].keys())
            sampLen = len(EMG[k1[0]][k2[0]][1])
            output = {}
            stats_out = {}
            bin_length = 500 # in ms
            bin_samps = int(bin_length * self.fs / 1000)
            num_bins = int(math.floor(sampLen / bin_samps))
            timepoints = np.linspace(-timeLag + (bin_length/1000),(sampLen/self.fs)-timeLag,num_bins)
            for task,channels in task_mapping.items():
                  target = channels
                  compares = [i for i in EMG if i not in target]
                  to = {}
                  stats_t ={}
                  for t in target:
                        tdata = [i+1 for i in EMG[t][task][2]]
                        co = {}
                        stats_c = {}
                        for c in compares:
                              # isolation comparison functionality. 
                              cdata = [i+1 for i in EMG[c][task][2]]
                              x = binned_muscle_isolation(cdata,tdata,num_bins=num_bins) 
                              co[c] = x
                              stats_c[c] = binned_timeseries_stats(x)
                        to[t] = co
                        output[t] = co
                        stats_out[t] = stats_c
                  # output[task] = to
            if plot or save or (plot==save==True):
                  k = 'EMG'
                  numChan = len(output)
                  nCol = 1
                  fullArr = math.ceil(numChan/nCol) * nCol
                  plotShape = np.shape(np.zeros(fullArr).reshape((-1,nCol)))
                  title = f'{subject}_{k} isolation index'
                  figname = f'{subject}_{k}_isolation_index'
                  fig, axs = plt.subplots(plotShape[0],plotShape[1],num =figname,sharex=True,sharey=True,figsize=(20,15))
                  for i,(target_mus,ax) in enumerate(zip(output,fig.axes)):                        
                        for j,(k,v) in enumerate(output[target_mus].items()):
                              stats = stats_out[target_mus][k].pvalue
                              stats = stats * len(stats)
                              statsPlot = [[x,1.1 + 0.15*j] for x,y in zip(timepoints,stats) if y <= 0.05]
                              statsPlot = np.array(statsPlot).reshape(-1,2).T
                              ax.errorbar(timepoints,np.mean(v,axis=0),yerr=np.std(v,axis=0),label=k,capsize=5,color=self.cs[j])
                              ax.scatter(timepoints,np.mean(v,axis=0),label='_',color=self.cs[j])
                              ax.scatter(statsPlot[0],statsPlot[1],label='_', color=self.cs[j], marker='*')
                              # ax.set_ylim(0,1.5)
                              ax.legend()
                        ax.set_title(f'{target_mus} isolation index')
                  fig.suptitle(title)
            if save:
                  plt.figure(fig)
                  plt.savefig(figPath / f'{figname}.svg')
                  with open(dataPath/ f'{figname}.pkl', 'wb') as fp:
                        pickle.dump(output,fp)
            
            if not plot:
                  plt.close(fig=fig)
            
            return output, stats_out
      
def binned_timeseries_stats(data):
      res = st.ttest_1samp(data,popmean=0,axis=0)
      return res
      

def binned_muscle_isolation(antagonist,target,num_bins):
      x = np.ones(np.shape(target)) - np.divide(antagonist,target) # 1 when target muscle is solely responsive 
      out = []
      for i in range(len(x)):
            dat = x[i][:]
            binned_res = np.array_split(dat,num_bins)
            res = np.array([np.average(j) for j in binned_res])
            res = np.where(res<0,0,res) # if antagonist muscle activity exceeds target muscle, set isolation index to zero.        
            out.append(res)
      return out
def binned_ERP(ERP, num_bins):
      out = []
      for i in range(len(ERP)):
            dat = ERP[i][:]
            binned_res = np.array_split(dat,num_bins)
            res = np.array([np.average(j) for j in binned_res])
            out.append(res)
      return out
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

      def getChannelOutcomes(self,significant: bool=False, alpha = 0.05):
            '''This function retrieves channel outcomes based on significance level and alpha value,
            organizing the results into DataFrames for d and rsq values.
            
            Parameters
            ----------
            significant : bool, optional
                  The `significant` parameter is a boolean flag that determines whether to filter the results
            based on statistical significance. If `significant` is set to True, the function will only
            consider channels with a p-value less than or equal to the specified alpha level. If
            `significant` is set to False, all
            alpha
                  The `alpha` parameter in the `getChannelOutcomes` function represents the significance level
            used for hypothesis testing. It is typically set to a value between 0 and 1, such as 0.05, to
            determine the threshold for statistical significance. 
            
            Returns
            -------
                  The function `getChannelOutcomes` returns two DataFrames, `dfd` and `dfrsq`, which contain
            the outcomes for each channel in terms of 'd' and 'rsq' values respectively.
            
            '''
            if significant:
                  channels = self.taskRes.loc[self.taskRes['p']<= alpha]['channel'].to_list()
            else:
                  channels = self.taskRes['channel'].to_list()
            channels = set(channels)
            temp = self.taskRes.loc[self.taskRes['channel'].isin(channels)]
            out_r, out_d = {},{}
            for c in channels:
                  data = temp.loc[temp['channel']==c]
                  data_r = {k:v for k,v in zip(data['task'],data['rsq'].to_numpy())}
                  data_d = {k:v for k,v in zip(data['task'],data['d'].to_numpy())}
                  out_d[c] =data_d
                  out_r[c] =data_r
            dfrsq = pd.DataFrame.from_dict(out_r,orient='index')
            dfd = pd.DataFrame.from_dict(out_d,orient='index')
            return dfd, dfrsq

      def evaluateClustering(self,data,clusterList:list,randomseed: int,title='PCA'):
            randomseed = random.seed(randomseed)
            plotData = data.T            
            for cluster in clusterList:
                  colors = distinctipy.get_colors(cluster,rng=randomseed)
                  fig, (ax1,ax2)=plt.subplots(1,2, num = f'{title} \nnum clusters={cluster}')
                  fig.suptitle(title)
                  ax1.set_ylim([0,len(data)+(cluster+1)*10])
                  ax1.set_xlim([-0.2,1])
                  classif = self.kmeans_cluster(data,cluster)
                  labels=classif.fit_predict(data)
                  silhouette_score = metrics.silhouette_score(data,labels)
                  silhouette_score_samples = metrics.silhouette_samples(data,labels)
                  y_lower = 10
                  for i in range(cluster):
                        local_cluster_vals = silhouette_score_samples[labels==i]
                        local_cluster_vals.sort()
                        size_cluster_i = local_cluster_vals.shape[0]
                        y_upper = y_lower + size_cluster_i
                        ax1.fill_betweenx(np.arange(y_lower,y_upper),0,local_cluster_vals,facecolor=colors[i],edgecolor=colors[i],alpha=0.7)
                        
                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                        # Compute the new y_lower for next plot
                        y_lower = y_upper + 10  # 10 for the 0 samples
                        cmap = getColorMatrix(colors,classif.labels_)
                        ax2.scatter(plotData[0],plotData[1],color=cmap)
                  ax1.set_title("The silhouette plot for the various clusters.")
                  ax1.set_xlabel("The silhouette coefficient values")
                  ax1.set_ylabel("Cluster label")

                  # The vertical line for average silhouette score of all the values
                  ax1.axvline(x=silhouette_score, color="red", linestyle="--")

                  ax1.set_yticks([])  # Clear the yaxis labels / ticks
                  # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                  
      def clusterChannels(self,significant: bool=False, alpha = 0.05):
            '''This function performs clustering on channel outcomes using PCA and KernelPCA, and visualizes
            the results in a plot.
            
            Parameters
            ----------
            significant : bool, optional
                  The `significant` parameter in the `clusterChannels` method is a boolean parameter with a
            default value of `False`. It is used to determine whether to consider only significant
            outcomes when clustering the channels.
            alpha
                  The `alpha` parameter in the `clusterChannels` method is a significance level used for
            hypothesis testing. It is typically set to a value between 0 and 1, representing the
            probability of rejecting the null hypothesis when it is actually true. 
            
            Returns
            -------
                  the value 0.
            
            '''
            randomseed = random.seed(35)
            dfd, dfrsq = self.getChannelOutcomes(significant,alpha)
            dfd[dfd.columns] = preprocessing.StandardScaler().fit_transform(dfd)
            n_comps = 2
            
            pca = decomposition.PCA(n_components=n_comps,random_state=randomseed)
            kspca = decomposition.KernelPCA(n_components=n_comps,kernel='sigmoid',random_state=randomseed)
            kppca = decomposition.KernelPCA(n_components=n_comps,kernel='rbf',random_state=randomseed)
            # TODO: check variance explained in each PC
            pca_res = pca.fit_transform(dfd)
            kspca_res = kspca.fit_transform(dfd)
            kppca_res = kppca.fit_transform(dfd)
            pcaDf = pd.DataFrame(abs(pca.components_),columns=dfd.columns,index=[f'PC{i+1}' for i in range(n_comps)])
            self.evaluateClustering(kspca_res,[2,3,4,5,6],randomseed , title='sigmoid kernel PCA')
            self.evaluateClustering(pca_res,  [2,3,4,5,6],randomseed, title='linear PCA')
            self.evaluateClustering(kppca_res,[2,3,4,5,6],randomseed  , title='RBF kernel PCA')
            
            n_clusters = 5
            res_colors = distinctipy.get_colors(n_clusters,rng=random.seed(35))
            cluster_res = self.kmeans_cluster(pca_res,num_clusters=n_clusters)
            cmap = getColorMatrix(res_colors,cluster_res.labels_)
            kcluster_res = self.kmeans_cluster(kspca_res,num_clusters=n_clusters)
            kcmap = getColorMatrix(res_colors,kcluster_res.labels_)
            
            
            """ Result Plotting"""
            fig = plt.figure()
            r,k = pca_res.T, kspca_res.T
            if n_comps < 3:
                  ax = fig.add_subplot(1,2,1 )
                  ax2 = fig.add_subplot(1,2,2)
                  ax. scatter(r[0],r[1],c=cmap,alpha=0.4)
                  ax2.scatter(k[0],k[1],c=kcmap,alpha=0.4)
            
            else:     
                  ax = fig.add_subplot(1,2,1,projection='3d')
                  ax2 = fig.add_subplot(1,2,2,projection='3d')
                  axLine = np.linspace(-1,1,10)
                  offAx = np.linspace(0,0,10)
                  ax.plot(xs=axLine,ys=offAx, zs=offAx, c=(0,0,0))
                  ax.plot(xs=offAx, ys=axLine,zs=offAx, c=(0,0,0))
                  ax.plot(xs=offAx, ys=offAx, zs=axLine, c=(0,0,0))
                  
                  ax2.plot(xs=axLine,ys=offAx, zs=offAx, c=(0,0,0))
                  ax2.plot(xs=offAx, ys=axLine,zs=offAx, c=(0,0,0))
                  ax2.plot(xs=offAx, ys=offAx, zs=axLine, c=(0,0,0))
                  ax. scatter(r[0],r[1],r[-1],c=cmap,alpha=0.4)
                  ax2.scatter(k[0],k[1],k[-1],c=kcmap,alpha=0.4)
            plt.show()
            return 0

      def kmeans_cluster(self,data:np.ndarray,num_clusters:int)->cluster.KMeans:
            classif = cluster.KMeans(n_clusters=num_clusters,random_state=0, n_init='auto')
            classif.fit(data)
            return classif
            


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
            
def getColorMatrix(colors, IndexVector):
      IndexVector = IndexVector - min(IndexVector) #make sure is 0 indexed. 
      return [colors[j] for j in IndexVector]

def export_ERP_Obj(obj, subject,fpath:Path|str,method):
            if isinstance(fpath,str):
                  fpath = Path(fpath)
            fp = fpath / f'{subject}_{method}.pkl'
            with open(fp, 'wb') as fp:
                  pickle.dump(obj,fp)
def load_ERP_Obj(fpath):
      with open(fpath, 'rb') as fp:
            x = pickle.load(fp)
      return x             