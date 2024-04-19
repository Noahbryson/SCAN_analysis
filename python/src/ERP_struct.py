import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from functions.stat_methods import confidence_interval
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
