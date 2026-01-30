import os
import csv
from pathlib import Path
import scipy.io as scio
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from .functions.filters import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patch
from matplotlib.axes import Axes
from typing import List
import seaborn as sns
import distinctipy
import math
import pickle
import seaborn as sns
from src.functions.stat_methods import paired_two_sample, nonlinear_fit_permutation_test,pdist2
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain
# from sklearn import metrics
# from sklearn.cluster import KMeans
from .functions.stat_methods import euclidean_distance
from .modules.response_datastructs import ERP_struct, plot_range_on_curve

class SCAN_group_analysis():
      def __init__(self, dataDir:Path, subjectList: list):
            self.path = dataDir
            self.subjects = subjectList
            self.subjectDirs = {sub:dataDir/sub for sub in subjectList}
            self.colorPalletBest = [(62/255,108/255,179/255), (27/255,196/255,225/255), (129/255,199/255,238/255),(44/255,184/255,149/255),(0,129/255,145/255), (193/255,189/255,47/255),(200/255,200/255,200/255)]
            self.movements = ['Hand','Foot','Tongue']
            

      def Load_all_ERPs(self,ERP_type:str='gamma')->None:
            self.ERPs = {}
            for k in self.subjectDirs:
                  self.ERPs[k] = self.load_ERP(k,ERP_type)
            
      def load_ERP(self,sessionKey: str,ERP_type:str='gamma')->ERP_struct:
            dir_contents = os.listdir(self.subjectDirs[sessionKey]/'ERPs')
            loc = next((i for i, e in enumerate(dir_contents) if ERP_type in e), len(dir_contents) - 1)
            fpath = self.subjectDirs[sessionKey]/'ERPs'/dir_contents[loc]
            with open(fpath, 'rb') as fp:
                  x:ERP_struct = pickle.load(fp)
            self.ERP_type = ERP_type
            return x
      def load_ERP_subset(self,sessionKey: list,ERP_type:str='gamma')->ERP_struct:
            out = {}
            for key in sessionKey:
                  out[key] = self.load_ERP(key,ERP_type)
            
            return out      
      
      def compare_EMG_isolation(self,session1,session2)->Figure:
            if not hasattr(self,'EMG_isolation'):
                  self.load_EMG_isolation()
            baseline = self.EMG_isolation[session1]
            compare = self.EMG_isolation[session2]
            rows = len(baseline)
            cols = 4
            colors = {i:j for i,j in zip(baseline,self.colorPalletBest[1:])}
            fig, ax = plt.subplots(rows,cols, sharex=True,sharey=True)
            for i,b in enumerate(baseline):
                  for j,muscle in enumerate(baseline[b]):
                        c = colors[muscle]
                        b1 = np.array(baseline[b][muscle])
                        c1 = np.array(compare[b][muscle])
                        delta_I = c1-b1
                        I_avg = np.average(delta_I,axis=0)
                        t = np.linspace(-0.5,5,delta_I.shape[-1])
                        res = stats.ttest_ind(b1,c1,axis=0)
                        statsPlot = np.array([[x,1+j*0.1] for x,y in zip(t,res.pvalue) if y <= 0.05]).reshape(-1,2).T
                        ax[i][0].scatter( t,I_avg,color=c,label=muscle)
                        ax[i][0].scatter( statsPlot[0],statsPlot[1], color=c, marker='*',label='_')
                        ax[i][0].errorbar(t,I_avg,yerr=np.std(delta_I,axis=0),color=c,label='_',capsize=4)
                        ax[i][1].errorbar(t,np.average(b1,axis=0),yerr=np.std(b1,axis=0),color=c,label='_',capsize=4)
                        ax[i][2].errorbar(t,np.average(c1,axis=0),yerr=np.std(c1,axis=0),color=c,label='_',capsize=4)
                        ax[i][1].scatter(t,np.average(b1,axis=0),color=c,label=muscle)
                        ax[i][2].scatter(t,np.average(c1,axis=0),color=c,label=muscle)
                        circle = patch.Circle((0,0.5-j),radius=0.4,color=c)
                        ax[i][3].add_patch(circle)
                        ax[i][3].annotate(muscle,(1,0.5-j))
                        # ax[i][3].set_xlim(-0.1,1)
                        # ax[i][3].set_ylim(-1,1)
                  ax[i][0].set_ylabel('Change in Isolation Index')
                  ax[i][1].set_ylabel('Isolation Index')
                  ax[i][2].set_ylabel('Isolation Index')
                  ax[i][0].set_title(f'{b} Change in Isolation Index')
                  ax[i][1].set_title(f'{b} Pre-RFA Isolation Index')
                  ax[i][2].set_title(f'{b} Post-RFA Isolation Index')
                  ax[i][3].set_title('Color Legend')
            ax[-1][0].set_xlabel('Time (s)')
            ax[-1][2].set_xlabel('Time (s)')
            ax[-1][1].set_xlabel('Time (s)')
            return fig           
            
      def load_EMG_isolation(self)-> None:
            self.EMG_isolation = {}
            for k,v in self.subjectDirs.items():
                  dir_contents = os.listdir(v)
                  loc = next((i for i, e in enumerate(dir_contents) if 'EMG_isolation' in e), len(dir_contents) - 1)
                  fpath = v/dir_contents[loc]
                  with open(fpath, 'rb') as fp:
                        x = pickle.load(fp)
                  self.EMG_isolation[k] = x
                  
      def load_rsq(self,reference_session) -> pd.DataFrame:
            """
            load_rsq _summary_

            Returns:
                  _type_: _description_
            """
            rsqs = pd.DataFrame()
            for k,v in self.subjectDirs.items():
                  df = pd.read_csv(v/f'{k}_rsq.csv')
                  df=df.set_index('channel')
                  df['session'] = k
                  rsqs = pd.concat([rsqs,df])
                  
            rsqs.reset_index(inplace=True)
            channel_labels, _ = self.load_channnel_labels(reference_session)
            data = rsqs.merge(channel_labels,on='channel')
            data['class']=data['class'].str.replace('face','tongue')
            return data 
      
      
      def load_latencies(self)->pd.DataFrame:
            output=pd.DataFrame()
            for k,v in self.subjectDirs.items():
                  df = pd.read_json(v/f'{k}_movement-latencies.json')
                  df['session'] = k
                  output = pd.concat([output,df])
                  
            output.reset_index(inplace=True)
            return output.drop('index',axis=1)
            
            
             
      def load_channnel_labels(self,reference_session)-> tuple:
            p = self.subjectDirs[reference_session]
            labels = pd.read_csv(p/'channel_classifications.csv')
            labels = labels.astype({'significant':'bool'})
            classes = sorted(list(set(labels['class'])))
            return labels, classes
      
      
            
      
      def compare_shared_rep(self,reference_session,compare_session,significant=True,region_search_strs=[],paired=False,channel_markers=[])->pd.DataFrame:
            """
            
            """
            
            data = self.load_rsq(reference_session)
            channel_labels, chan_classes = self.load_channnel_labels(reference_session)
            regions=set(channel_labels['region'].to_list())
            data['shared_rep'] = data.apply(lambda x: shared_rep(x['Hand'],x['Tongue'],x['Foot']),axis=1)
            
            chan_classes = np.unique(data['class'])
            nonspec = [i for i in chan_classes if i.find('-')>-1]
            chan_classes = [i for i in chan_classes if i.find('-')==-1]
            chan_classes.append('non-specific')
            locs = data['class'].isin(nonspec)
            data.loc[locs,'class'] = 'non-specific'
            if len(region_search_strs) > 0:
                  subregions = []
                  if isinstance(region_search_strs,str):
                        region_search_strs = [region_search_strs]
                  for i in region_search_strs:
                        subregions.extend([s for s in regions if i in s])
                  subregions = np.unique(subregions).tolist()
                  data['target_region'] = data['region'].apply(lambda x: x in subregions)
                  data = data.loc[data['target_region']==True]
            if significant:
                  data = data.loc[data['significant']==True]
                  
            if paired:
                  fig,axs = plt.subplots(1,len(chan_classes),sharex=False,sharey=True)
                  stats = {}
                  rm_cols = []
                  for ax,i in zip(axs,chan_classes):
                        df = data.loc[data['class'] == i]
                        if len(df) > 0:
                              a = df.loc[data['session'] == reference_session]
                              b = df.loc[data['session'] == compare_session]
                              
                              res = paired_two_sample(a['shared_rep'].to_numpy(),b['shared_rep'].to_numpy(),ax,plot_mean=False)
                              a_sub = a[a['channel'].isin(channel_markers)]
                              b_sub = b[b['channel'].isin(channel_markers)]
                              ax.scatter([0 for _ in range(len(a_sub))],a_sub['shared_rep'].to_numpy(),marker='*',color=(1,1,0))
                              ax.scatter([1 for _ in range(len(b_sub))],b_sub['shared_rep'].to_numpy(),marker='*',color=(0,1,1))
                              stats[i] = res[0]
                              ax.set_xticks([0,1])        
                              ax.set_xticklabels([reference_session,compare_session])
                              ax.set_title(i)   
                              if res[0].pvalue < 0.05:
                                    ax.text(.4,1.1*np.max([data['shared_rep']]),'*')     
                        else:
                              print(f'no {i} tuned channels')
                              rm_cols.append(ax)   
                  axs[0].set_ylabel(f'Shared Representation Magnitude')
                  for a in axs.flat:
                        a.label_outer()
                        a.set_ylim([-0.5,1])
                        a.tick_params(axis='x', labelrotation=45)
                  for i in rm_cols:
                        fig.delaxes(i)
            else:      
                  fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
                  sns.stripplot(data=data,x='class',y='shared_rep',hue='session',ax=ax)      
            data_clusters = cluster_rsq_by_label(data,significant)
            fig.suptitle('Change in Shared Representation')
            return data
      
      def compare_tuning(self,reference_session,compare_session,significant=True,region_search_strs=[],paired=False,effect='Magnitude')->None:
            
            rsqs = self.load_rsq()
            channel_labels, chan_classes = self.load_channnel_labels(reference_session)
            regions=set(channel_labels['region'].to_list())
            data = rsqs.merge(channel_labels,on='channel')
            data[['tuning','Magnitude','Angle']] = data.apply(lambda x: tuning(x['Hand'],x['Tongue'],x['Foot']),axis=1,result_type='expand')
            data['Magnitude'] = data['Magnitude'].astype('float').copy()
            data['Angle'] = data['Angle'].astype('float').copy()
            if len(region_search_strs) > 0:
                  subregions = []
                  if isinstance(region_search_strs,str):
                        region_search_strs = [region_search_strs]
                  for i in region_search_strs:
                        subregions.extend([s for s in regions if i in s])
                  subregions = np.unique(subregions).tolist()
                  data['target_region'] = data['region'].apply(lambda x: x in subregions)
                  data = data.loc[data['target_region']==True]
            if significant:
                  data = data.loc[data['significant']==True]
                  
            if paired:
                  fig,axs = plt.subplots(1,len(chan_classes),sharex=False,sharey=True)
                  stats = {}
                  rm_cols = []
                  for ax,i in zip(axs,chan_classes):
                        df = data.loc[data['class'] == i]
                        if len(df) > 0:
                              a = df.loc[data['session'] == reference_session]
                              b = df.loc[data['session'] == compare_session]
                              
                              res,ax = paired_two_sample(a[effect].to_numpy(),b[effect].to_numpy(),ax)
                              stats[i] = res
                              ax.set_xticks([0,1])        
                              ax.set_xticklabels([reference_session,compare_session])
                              statstr = f'{i}\nW={res.statistic}\np={round(res.pvalue,4)}'
                              ax.set_title(statstr)        
                              # ax.text(0,np.max([a[effect]]),statstr)
                              if res.pvalue < 0.05:
                                    ax.text(0,1.1*np.max([a[effect]]),'*')
                        else:
                              print(f'no {i} tuned channels')
                              rm_cols.append(ax)
                  axs[0].set_ylabel(f'Somatotopic Tuning {effect}')
                  for a in axs.flat:
                        a.label_outer()
                        if effect == 'Angle':
                              a.set_ylim([0,360])
                        else:
                              a.set_ylim([-0.5,1])
                        a.tick_params(axis='x', labelrotation=45)
                  for i in rm_cols:
                        fig.delaxes(i)
            else:      
                  fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
                  sns.stripplot(data=data,x='class',y='shared_rep',hue='session',ax=ax)      
            data_clusters = cluster_rsq_by_label(data,significant)      
            fig.suptitle(f'Somatotopic Tuning {effect}')
            
            
            
      def compare_rsq(self,reference_session,compare_session,significant=True,region_search_strs=[],paired=False)->Figure:
            
            rsqs = self.load_rsq()
            channel_labels, chan_classes = self.load_channnel_labels(reference_session)
            regions=set(channel_labels['region'].to_list())
            data = rsqs.merge(channel_labels,on='channel')
            if len(region_search_strs) > 0:
                  subregions = []
                  if isinstance(region_search_strs,str):
                        region_search_strs = [region_search_strs]
                  for i in region_search_strs:
                        subregions.extend([s for s in regions if i in s])
                  subregions = np.unique(subregions).tolist()
                  data['target_region'] = data['region'].apply(lambda x: x in subregions)
                  data = data.loc[data['target_region']==True]
            if significant:
                  data = data.loc[data['significant']==True]
            
            if paired:
                  fig,axs = plt.subplots(3,len(chan_classes),sharex=False,sharey=True)
                  stats = {}
                  rm_cols = []
                  for ax,i in zip(axs.T,chan_classes):
                        df = data.loc[data['class'] == i]
                        if len(df) > 0:
                              a = df.loc[data['session'] == reference_session]
                              b = df.loc[data['session'] == compare_session]
                              for j,movement in enumerate(self.movements):
                                    res = paired_two_sample(a[movement].to_numpy(),b[movement].to_numpy(),ax[j])
                                    if i in stats:
                                          stats[i].update({movement:res})
                                    else:
                                          stats[i] = {movement:res}
                              ax[-1].set_xticks([0,1],[reference_session,compare_session])        
                              ax[-1].set_xticklabels([reference_session,compare_session])
                              ax[0].set_title(i)        
                        else:
                              print(f'no {i} tuned channels')
                              rm_cols.append(ax)
                  axs[0,0].set_ylabel(f'{self.movements[0]} r^2')
                  axs[1,0].set_ylabel(f'{self.movements[1]} r^2')
                  axs[2,0].set_ylabel(f'{self.movements[2]} r^2')
                  for a in axs.flat:
                        a.label_outer()
                        a.set_ylim([-0.5,1])
                        a.tick_params(axis='x', labelrotation=45)
                  for i in rm_cols:
                        for j in i:
                              fig.delaxes(j)

            else:
                  fig,axs = plt.subplots(3,1,sharex=True,sharey=True)
                  sns.stripplot(data=data,x='class',y='Hand',hue='session',ax=axs[0])      
                  sns.stripplot(data=data,x='class',y='Tongue',hue='session',ax=axs[1])      
                  sns.stripplot(data=data,x='class',y='Foot',hue='session',ax=axs[2])      
            # data_clusters = cluster_rsq_by_label(data,significant)
            fig.suptitle('Change in r^2 due to ablation')
            return fig
      
      
      def compare_ERP_pair(self,session1:str, session2:str,savePath:str|Path|None=None,plot:bool=True,fs:int=2000,method:str='cluster',stdErr: bool=True)-> tuple:
            from src.functions.stat_methods import cluster_permutation,binned_timeseries_compare
            save = True 
            if isinstance(savePath,type(None)):
                  save = False
            from src.functions.graphics import hex2RGB, stockRGB
            r,g,b = stockRGB()
            data = self.load_ERP_subset([session1,session2])
            results = {}
            trajectories = sorted(set(data[session1].mapping.keys()) & set(data[session2].mapping.keys()))
            # chans = sorted(list(set(data[session1].data) & set(data[session2].data)))
            fp = f'{savePath}_{session1}-vs-{session2}_ERP.pkl'
            if os.path.exists(fp):
                  with open(fp, 'rb') as f:
                        results = pickle.load(f) 
            else:      
                  for traj in trajectories:
                        chans = data[session1].mapping[traj]
                        for channel in chans:
                              temp = {}
                              for event in data[session1].data[channel]:
                                    a = data[session1].data[channel][event][-1]
                                    b = data[session2].data[channel][event][-1]
                                    # res = corr_permutation(np.array(a),np.array(b))
                                    #TODO: abandon this for now, we dont care enough about changes in morphology
                                    #     just go with binning the waveforms and doing 2-sample unpaired
                                    #     OR compare the distributions of broadband power during tasks and include that with the traces.
                                    #     could also use curve AUC and compare that with 2 sample to account for overall morphology. 
                                    match method:
                                          case 'cluster':
                                                res = cluster_permutation(np.array(a),np.array(b))
                                                temp[event] = res
                                          case 'bin':      
                                                signal_blocks = 15
                                                res,loc = binned_timeseries_compare(np.array(a),np.array(b),signal_blocks=signal_blocks)
                                                res.FDR_correction()
                                                temp[event] = [res,loc]
                                          case '_':
                                                import sys
                                                print('no available method specified, aborting')
                                                sys.exit()
                              results[channel] = temp
                              if save:
                                    with open(fp, 'wb') as f:
                                          pickle.dump(results,f) 
            if plot:
                  for traj in trajectories:
                        chans = data[session1].mapping[traj]
                        numChan = len(chans)
                        nCol = 4
                        fullArr = math.ceil(numChan/nCol) * nCol
                        plotShape = np.shape(np.zeros(fullArr).reshape((-1,nCol)))
                        figname = f'{traj} {session1} vs {session2} {self.ERP_type} Comparison'
                        fig, axs = plt.subplots(plotShape[0],plotShape[1],num=figname,sharex=True,sharey=True,figsize=(10,7))
                        ax: List[Axes] = axs.ravel()
                        from src.functions.graphics import stockRGB
                        colors = stockRGB()
                        for i,channel in enumerate(chans):
                              yshifts = [0,-1.5,-3]
                              ticks = [i.split('_')[-1] for i in data[session1].data[channel].keys()]
                              for j,event in enumerate(data[session1].data[channel]):
                                    yshift = yshifts[j]
                                    res = results[channel][event]
                                    a = data[session1].data[channel][event]
                                    b = data[session2].data[channel][event]
                                    t=np.linspace(0,len(a[0])/fs,len(a[0]))
                                    match method:
                                          case 'bin':
                                                ps = res[0].corrected_p
                                                stat_idxs = [idx for idx,i in enumerate(ps) if i <0.05]
                                                if len(stat_idxs) > 0:
                                                      stat_locs = [np.median(res[1][i]) for i in stat_idxs]
                                                      ys = np.max([np.max(a[0]),np.max(b[0])])
                                                      ys = [ys for _ in stat_locs]
                                                      text = ['*' for _ in stat_locs]
                                                      for x,y in zip(stat_locs,ys):
                                                            ax[i].text(x,y,'*')
                                          case 'cluster':
                                                for x,p in zip(res[1],res[2]):
                                                      x = x[0]
                                                      ax[i].plot(t[x],np.min([np.min(a[0]),np.min(b[0])])*np.ones(x.shape)+yshift,alpha=0.5,color=colors[1],lw=3)
                                                      if p < 0.05:
                                                            xloc = np.mean(t)
                                                            yloc = np.max([np.max(a[0]),np.max(b[0])])
                                                            ax[i].text(xloc,yloc,'*')
                                    
                                    
                                    if stdErr:
                                          a_dev = a[1]/np.sqrt(len(a[2]))
                                          b_dev = b[1]/np.sqrt(len(b[2]))
                                    else:
                                          a_dev = a[1]
                                          b_dev = b[1]
                                    plot_range_on_curve(t,a[0]+yshift,a_dev,ax=ax[i],color=colors[0])
                                    plot_range_on_curve(t,b[0]+yshift,b_dev,ax=ax[i],color=colors[2])
                                    ax[i].plot(t,b[0]+yshift,color=colors[2])
                                    ax[i].plot(t,a[0]+yshift,color=colors[0])
                                    ax[i].set_yticks(yshifts)
                                    ax[i].set_yticklabels(ticks)
                                    ax[i].set_title(channel)
                                    ax[i].vlines(0,ymin = -1,ymax=0,colors=(0,0,0))
                                          
                        for a in axs.flat:
                              a.label_outer()
                              a.set_ylim([np.min(yshifts)-np.abs(np.mean(np.diff(yshifts))),np.max(yshifts)+np.abs(np.mean(np.diff(yshifts)))])
                        if savePath:
                              outDir = f'{savePath}/png'
                              os.makedirs(outDir,exist_ok=True)
                              outDir = f'{savePath}/svg'
                              os.makedirs(outDir,exist_ok=True)
                              outDir = f'{savePath}/png/{session1}-vs-{session2}_{traj}.png'
                              plt.savefig(outDir)
                              outDir = f'{savePath}/svg/{session1}-vs-{session2}_{traj}.svg'
                              plt.savefig(outDir)
                              plt.close()         
                  
                  
                  
            return results
      
      
      
      def get_subject_session(self,target_session) -> pd.DataFrame:
            data = self.load_rsq(target_session)
            return data[data['session']==target_session]
      
      
def plot_latencies(data:pd.DataFrame)->Figure:
      import random
      fig, ax = plt.subplots(1,1)
      labels = np.unique(data['session'].to_list())
      n = len(labels)
      ax.spines[['right','top','bottom']].set_visible(False)
      cs = distinctipy.get_colors(3,pastel_factor=0.5,rng=random.seed(35))
      cs = [(0.97602050272426, 0.33490232967809724, 0.3437729866283861),(0.3630238841279352, 0.3377308638235165, 0.957096838172224)]
      
      pallete = {i:j for i,j in zip(labels,cs)}
      sns.boxplot(data,x='Movement',y='Latency',hue='session',ax=ax,palette=pallete)
      sns.swarmplot(data,x='Movement',y='Latency',hue='session',ax=ax,palette=pallete)
      ax.tick_params(direction='in')
      ax.set_ylim([0,2000])
      
      
      return fig
            

def ablation_effect(brain:PyBrain, effect:pd.DataFrame, effect_name:str,
      volume:np.ndarray, volumeLabel:str, ROIs: list,centroid: bool=True,plot: bool=True,ax: Axes|None=None)-> tuple:
            
      pre = effect.loc[effect['session'].str.find('pre') >= 0].set_index('channel')
      post = effect.loc[effect['session'].str.find('post') >= 0].set_index('channel')
      out_effect = f'delta {effect_name}'
      cols = pre.columns.to_list()[3:7]
      cols.append(pre.columns[8])
      df = pre[cols].copy()
      df.loc[:,out_effect] = post[effect_name] - pre[effect_name]
      _,_,ax = volume_effect_distance(brain, df.reset_index(), out_effect,volume, volumeLabel, ROIs,centroid,plot,ax=ax)
      if plot:
            ax.set_title(f'Change in {effect_name} due\nto distance from {volumeLabel}')
            # ax.set_xscale('log')

def volume_effect_distance(brain:PyBrain, effect:pd.DataFrame, effect_name:str,
      volume:np.ndarray, volumeLabel:str, ROIs: list,centroid: bool=False,plot: bool=False, ax: Axes|None=None) -> tuple:     
      
      if  not isinstance(ax,Axes) and plot:
            
            ax = plt.axes()
      idxs = effect['region'].apply(lambda x: np.any([1 for i in ROIs if x.find(i) > -1]))
      effectROI = effect.loc[idxs]
      res = effectROI.apply(lambda x: brain.electrodeNamesKey[x['channel']],axis=1)
      effectROI.loc[:,'vera'] = res
      res = effectROI['vera'].apply(lambda x: brain.electrodes.Location[brain.electrodes.Name.index(x)])
      effectROI.loc[:,'coords'] = res
      # chan_classes = np.unique(effectROI['class'])
      # nonspec = [i for i in chan_classes if i.find('-')>-1]
      # chan_classes = [i for i in chan_classes if i.find('-')==-1]
      # chan_classes.append('non-specific')
      # locs = effectROI['class'].isin(nonspec)
      # effectROI['class_old'] = effectROI['class']
      # effectROI.loc[locs,'class'] = 'non-specific'
      
      
      center = np.mean(volume,axis=0)
      centroids = {volumeLabel:center}
      
      tag = f'delta_r {volumeLabel}'
      if centroid:
            res = effectROI.apply(lambda x: euclidean_distance(x['coords'],center),axis=1)
            effectROI.loc[:,tag] = res
      else:
            res = pdist2(np.vstack(effectROI['coords'].to_numpy()),np.array(volume),num_mins=1)
            distances = res[:,0]
            mean_dists = np.mean(res,axis=1)
            # distances[np.where(mean_dists < 4)[0]] = 0
            effectROI.loc[:,tag] = distances
      classes = ['Hand','Foot','Tongue']
      popt,pconv = curve_fit(inverse_r2,effectROI[tag],effectROI[effect_name]) 
      res = nonlinear_fit_permutation_test(effectROI[tag],effectROI[effect_name],inverse_r2,popt)
      idx = effectROI['class'].isin([i.lower() for i in classes])
      subset = effectROI.loc[idx]
      popt,pcov = curve_fit(inverse_r2,subset[tag],subset[effect_name]) 
      res2 = nonlinear_fit_permutation_test(subset[tag],subset[tag],inverse_r2,popt)
      if plot:
            
            sns.scatterplot(data=effectROI,x=tag,y=effect_name,hue='class',ax=ax)
            r_vals = np.linspace(1,effectROI[tag].max(),100)
            rsq_decay = inverse_r2(r_vals,popt[0],popt[1])
            rsq_decay2 = inverse_r2(r_vals,popt[0],popt[1])
            ax.set_ylim([-0.5, 1])
            ax.plot(r_vals,rsq_decay)
            ax.plot(r_vals,rsq_decay2)
            ax.annotate(f'$R^2$={round(res.result,4)}\np={round(res.pvalue,5)}',(0.7,0.7),xycoords='axes fraction')
            ax.annotate(f'somatotopic $R^2$={round(res2.result,4)}\np={round(res2.pvalue,5)}',(0.7,0.4),xycoords='axes fraction')
            return centroids, effectROI, ax
      return centroids, effectROI



def compute_effect_centroid(brain:PyBrain, effect:pd.DataFrame, ROIs: list,plot: bool=False) -> tuple:
      
      # idxs = effect['region'].apply(lambda x: np.any([1 for i in ROIs if x.find(i) > -1]))
      effectROI = effect.loc[effect['region'].apply(lambda x: np.any([1 for i in ROIs if x.find(i) > -1]))].copy()
      # res = effectROI.apply(lambda x: brain.electrodeNamesKey[x['channel']],axis=1)
      effectROI.loc[:,'vera'] = effectROI.apply(lambda x: brain.electrodeNamesKey[x['channel']],axis=1).copy()
      # effectROI.loc[:,'vera'] = res
      # res = effectROI['vera'].apply(lambda x: brain.electrodes.Location[brain.electrodes.Name.index(x)])
      effectROI.loc[:,'coords'] = effectROI['vera'].apply(lambda x: brain.electrodes.Location[brain.electrodes.Name.index(x)]).copy()
      # effectROI.loc[:,'coords'] = res
      classes = ['Hand','Foot','Tongue']

      chan_classes = np.unique(effectROI['class'])
      nonspec = [i for i in chan_classes if i.find('-')>-1]
      chan_classes = [i for i in chan_classes if i.find('-')==-1]
      chan_classes.append('non-specific')
      locs = effectROI['class'].isin(nonspec)
      effectROI['class_old'] = effectROI['class'].copy()
      effectROI.loc[locs,'class'] = 'non-specific'

      centroids = {}
      
      fig,ax = plt.subplots(len(classes),1,sharex=True,sharey=True)
      for i,a in zip(classes,ax):
            subset = effectROI[effectROI['class'] == i.lower()]
            center = weighted_centroid(subset[i],subset['coords'])
            centroids[f'{i} centroid']=center
            
            res = effectROI.apply(lambda x: euclidean_distance(x['coords'],center),axis=1)
            # effectROI.loc[:,f'delta_r {i}'] = res
            
            effectROI.loc[:,f'delta_r {i}'] =  effectROI.apply(lambda x: euclidean_distance(x['coords'],center),axis=1)
            
            # effectROI.plot.scatter(x=f'delta_r {i}',y=i,ax=a,hue='class')
            import seaborn as sns
            
            # effectROI.plot.scatter(x=f'delta_r {i}',y=i,ax=a,hue='class')
            sns.scatterplot(data=effectROI,x=f'delta_r {i}',y=i,ax=a,hue='class')
            r_vals = np.linspace(1,effectROI[f'delta_r {i}'].max(),100)
            
            popt,pcov = curve_fit(inverse_r2,effectROI[f'delta_r {i}'],effectROI[i]) 
            rsq_decay = inverse_r2(r_vals,popt[0],popt[1])
            a.plot(r_vals,rsq_decay)
            
            
            
            # coeff_deter = compute_coeff_determ(effectROI[i],inverse_r2(effectROI[f'delta_r {i}'],p[0],p[1]))
            
            
            # poly = np.polynomial.Polynomial.fit(effectROI[f'delta_r {i}'],effectROI[i],deg=1)
            # y = poly(r_vals)
            # a.plot(r_vals,y)
            # p,v = curve_fit(inverse_r,effectROI[f'delta_r {i}'],effectROI[i]) 
            # rsq_decay = inverse_r(r_vals,p[0],p[1])
            # a.plot(r_vals,rsq_decay)
                        
            res = nonlinear_fit_permutation_test(effectROI[f'delta_r {i}'],effectROI[i],inverse_r2,popt)
            a.annotate(f'$R^2$={round(res.result,4)}\np={round(res.pvalue,5)}',(0.7,0.7),xycoords='axes fraction')
            idx = effectROI['class'].isin([i.lower() for i in classes])
            subset = effectROI.loc[idx]
            popt,pcov = curve_fit(inverse_r2,subset[f'delta_r {i}'],subset[i]) 
            rsq_decay = inverse_r2(r_vals,popt[0],popt[1])
            a.plot(r_vals,rsq_decay)
            res2 = nonlinear_fit_permutation_test(subset[f'delta_r {i}'],subset[i],inverse_r2,popt)
            a.annotate(f'somatotopic $R^2$={round(res2.result,4)}\np={round(res2.pvalue,5)}',(0.7,0.4),xycoords='axes fraction')
            a.set_ylim([-0.5, 1])
            a.set_xlim([0, 80])
      if not plot:
            plt.close(fig)
            return centroids,effectROI
      return centroids,effectROI,fig
      
      
def cluster_rsq_by_label(data,significant)-> None:
      clusterBois = {}
      
      for i in set(data['class']):
            pass
      
            
      return 0
def shared_rep(x,y,z) -> float:
      from math import sqrt 
      return sqrt(x**2+y**2+z**2)

def inverse_r2(r,a,c) -> float:
      return a / (r**2) + c

def inverse_r(r,a,c)->float:
      return a/r + c

def weighted_centroid(data,coords)->np.ndarray:
      test = np.histogram_bin_edges(data,'fd');hist = np.histogram(data,bins=test)
      weights = hist_weighting(data,hist)
      x = np.sum(np.multiply(weights,coords)) / np.sum(weights)
      print(x)
      print(np.mean(coords))
      return x

def hist_weighting(data,hist)->np.array:
      weights = [i+1 for i in range(len(hist[0]))]
      hist_centers = hist[1][1:] - np.diff(hist[1])/2
      out = np.zeros(np.shape(data))
      for i,v in enumerate(data):
            threshIdx = np.argmin(np.abs(hist_centers-v))
            out[i] = weights[threshIdx]
      return out

def tuning(hand,foot,tongue)-> tuple:
      from src.SCAN_SingleSessionAnalysis import complex_angle
      hand_c =  np.exp(1j*np.pi/6)   * hand
      foot_c =  np.exp(1j*np.pi*3/2) * foot
      tongue_c =np.exp(1j*np.pi*5/6) * tongue
      res = hand_c+foot_c+tongue_c
      mag = float(abs(res))
      theta = complex_angle(res)
      return res,mag,theta
