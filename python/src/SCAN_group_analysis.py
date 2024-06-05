import os
import csv
from pathlib import Path
import scipy.io as scio
import scipy.stats as stats
import pandas as pd
from functions.filters import *
import distinctipy
import math
import pickle
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from functions.stat_methods import mannwhitneyU, cohendsD, calc_ROC, geometric_mean,kde,euclidean_distance, signed_cross_correlation, angle_3D, angle_distances_3D
from modules.response_datastructs import ERP_struct, export_ERP_Obj, load_ERP_Obj

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
            cols = 3
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
            #       ax[i][0].set_ylabel('Change in Isolation Index')
            #       ax[i][1].set_ylabel('Isolation Index')
            #       ax[i][2].set_ylabel('Isolation Index')
            #       ax[i][0].set_title(f'{b} Change in Isolation Index')
            #       ax[i][1].set_title(f'{b} Pre-RFA Isolation Index')
            #       ax[i][2].set_title(f'{b} Post-RFA Isolation Index')
            # ax[-1][0].set_xlabel('Time (s)')
            # ax[-1][2].set_xlabel('Time (s)')
            # ax[-1][1].set_xlabel('Time (s)')
            # for a in ax.flat:
            #       a.legend()
            
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



if __name__ == "__main__":
      subjects = ['BJH041_pre_ablation','BJH041_post_ablation']
      import platform
      localEnv = platform.system()
      userPath = Path(os.path.expanduser('~'))
      if localEnv == 'Windows':
            dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
      else:
            dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"
      a = SCAN_group_analysis(dataPath/'Aggregate',subjects)
      a.compare_EMG_isolation(subjects[0],subjects[1])
      a.LoadERPs()
      plt.show()




