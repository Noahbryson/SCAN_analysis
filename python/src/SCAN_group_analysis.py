import os
import csv
from pathlib import Path
import scipy.io as scio
import scipy.stats as stats
import pandas as pd
from .functions.filters import *
import matplotlib.patches as patch
import seaborn as sns
import distinctipy
import math
import pickle
import seaborn as sns

# from sklearn import metrics
# from sklearn.cluster import KMeans
# from .functions.stat_methods import mannwhitneyU, cohendsD, calc_ROC, geometric_mean,kde,euclidean_distance, signed_cross_correlation, angle_3D, angle_distances_3D
from .modules.response_datastructs import ERP_struct, export_ERP_Obj, load_ERP_Obj

class SCAN_group_analysis():
      def __init__(self, dataDir:Path, subjectList: list):
            self.path = dataDir
            self.subjects = subjectList
            self.subjectDirs = {sub:dataDir/sub for sub in subjectList}
            self.colorPalletBest = [(62/255,108/255,179/255), (27/255,196/255,225/255), (129/255,199/255,238/255),(44/255,184/255,149/255),(0,129/255,145/255), (193/255,189/255,47/255),(200/255,200/255,200/255)]

            
            
            
            
      
      def LoadERPs(self,erp_type:str='gamma'):
            self.ERPs = {}
            for k,v in self.subjectDirs.items():
                  dir_contents = os.listdir(v/'ERPs')
                  loc = next((i for i, e in enumerate(dir_contents) if erp_type in e), len(dir_contents) - 1)
                  fpath = v/'ERPs'/dir_contents[loc]
                  with open(fpath, 'rb') as fp:
                        x:ERP_struct = pickle.load(fp)
                  self.ERPs[k] = x
                  
      def compare_EMG_isolation(self,session1,session2):
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
            # for a in ax.flat:
                  # a.legend()
            
            return 0            
            
      def load_EMG_isolation(self):
            self.EMG_isolation = {}
            for k,v in self.subjectDirs.items():
                  dir_contents = os.listdir(v)
                  loc = next((i for i, e in enumerate(dir_contents) if 'EMG_isolation' in e), len(dir_contents) - 1)
                  fpath = v/dir_contents[loc]
                  with open(fpath, 'rb') as fp:
                        x = pickle.load(fp)
                  self.EMG_isolation[k] = x
                  
      def __load_rsq(self):
            rsqs = pd.DataFrame()
            for k,v in self.subjectDirs.items():
                  df = pd.read_csv(v/f'{k}_rsq.csv')
                  df=df.set_index('channel')
                  df['session'] = k
                  rsqs = pd.concat([rsqs,df])
            return rsqs.reset_index()      
      def __load_channnel_labels(self,reference_session):
            p = self.subjectDirs[reference_session]
            labels = pd.read_csv(p/'channel_classifications.csv')
            self.labels = labels.astype({'significant':'bool'})
            return labels
      
      def compare_shared_rep(self,reference_session,significant=True,region_search_strs=[]):
            
            rsqs = self.__load_rsq()
            channel_labels = self.__load_channnel_labels(reference_session)
            regions=set(channel_labels['region'].to_list())
            data = rsqs.merge(channel_labels,on='channel')
            data['shared_rep'] = data.apply(lambda x: shared_rep(x['Hand'],x['Tongue'],x['Foot']),axis=1)
            if len(region_search_strs) > 0:
                  subregions = []
                  for i in region_search_strs:
                        subregions.extend([s for s in regions if region_search_strs in s])
                  subregions = np.unique(subregions).tolist()
                  data['target_region'] = data['region'].apply(lambda x: x in subregions)
                  data = data.loc[data['target_region']==True]
            if significant:
                  data = data.loc[data['significant']==True]
            fig,ax = plt.subplots(1,1,sharex=True,sharey=True)

            # ax[0]=sns.stripplot(data=data,x='class',y='Hand',hue='session')      
            # ax[1]=sns.stripplot(data=data,x='class',y='Tongue',hue='session')      
            # ax[2]=sns.stripplot(data=data,x='class',y='Foot',hue='session')      
            sns.swarmplot(data=data,x='class',y='shared_rep',hue='session',ax=ax)      
            data_clusters = cluster_rsq_by_label(data,significant)
      def compare_rsq(self,reference_session,significant=True,region_search_strs=[]):
            rsqs = self.__load_rsq()
            channel_labels = self.__load_channnel_labels(reference_session)
            regions=set(channel_labels['region'].to_list())
            data = rsqs.merge(channel_labels,on='channel')
            if len(region_search_strs) > 0:
                  subregions = []
                  for i in region_search_strs:
                        subregions.extend([s for s in regions if region_search_strs in s])
                  subregions = np.unique(subregions).tolist()
                  data['target_region'] = data['region'].apply(lambda x: x in subregions)
                  data = data.loc[data['target_region']==True]
            if significant:
                  data = data.loc[data['significant']==True]
            fig,ax = plt.subplots(3,1,sharex=True,sharey=True)

            # ax[0]=sns.stripplot(data=data,x='class',y='Hand',hue='session')      
            # ax[1]=sns.stripplot(data=data,x='class',y='Tongue',hue='session')      
            # ax[2]=sns.stripplot(data=data,x='class',y='Foot',hue='session')      
            sns.swarmplot(data=data,x='class',y='Hand',hue='session',ax=ax[0])      
            sns.swarmplot(data=data,x='class',y='Tongue',hue='session',ax=ax[1])      
            sns.swarmplot(data=data,x='class',y='Foot',hue='session',ax=ax[2])      
            data_clusters = cluster_rsq_by_label(data,significant)
            
            return 0
def cluster_rsq_by_label(data,significant):
      clusterBois = {}
      
      for i in set(data['class']):
            pass
            
      return 0
      
def shared_rep(x,y,z):
      from math import cbrt
      return cbrt(x*y*z)


