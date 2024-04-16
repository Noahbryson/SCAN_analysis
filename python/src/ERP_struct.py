import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from functions.stat_methods import confidence_interval
import math

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
            self.trajectories = set([i[0:2] for i in data['1_Hand'][0].keys()])
            channels = [i for i in data['1_Hand'][0].keys()]
            self.mapping = {}
            for i in self.trajectories:
                  self.mapping[i] = [j for j in channels if j.find(i)>=0]
            self.num_traj = len(self.trajectories)
            self.data,self.datalabels= self._reshape_data(data,channels)            
      def averageERP(self):
            figs = []
            for i in range(self.num_traj):
                  traj = self.trajectories[i]
                  channels = self.mapping[traj]
                  "https://stackoverflow.com/questions/41176248/reshape-arbitrary-length-vector-into-square-matrix-with-padding-in-numpy"
                  #TODO: take channel listing at reshape to nearest square/rectangle to map channels nicely for the subplotting function.
                  fig, ax = plt.subplots()
            return 0
      
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
      
      
      def plotAverages_per_trajectory(self,subject):
            figs = []
            for k,v in self.mapping.items():
                  numChan = len(v)
                  nCol = 4
                  fullArr = math.ceil(numChan/nCol) * nCol
                  plotShape = np.shape(np.zeros(fullArr).reshape((-1,nCol)))
                  
                  data = [self.data[x] for x in v]
                  title = f'{subject} {k}, {self.method}, {self.filterband} Hz ERPs'
                  fig, axs = plt.subplots(plotShape[0],plotShape[1],num = title,sharey=True)
                  # for a in axs.flat:
                  #       a.set_visible(False)
                  for i,(vals,ax) in enumerate(zip(data,fig.axes)):
                        keys = list(vals.keys())
                        cs = [(0.2,0.4,0.9),(0.4,0.9,0.2),(0.9,0.1,0.4)]
                        t = np.linspace(0,len(vals[keys[0]][0])/self.fs,len(vals[keys[0]][0]))                        
                        # a.set_visible(True)
                        ax.plot(t,vals[keys[0]][0], label=keys[0],color=cs[0])
                        ax.plot(t,vals[keys[1]][0], label=keys[1],color=cs[1])
                        ax.plot(t,vals[keys[2]][0], label=keys[2],color=cs[2])
                        plot_range_on_curve(t,vals[keys[0]][0],vals[keys[0]][1],ax,color=cs[0])
                        plot_range_on_curve(t,vals[keys[1]][0],vals[keys[1]][1],ax,color=cs[1])
                        plot_range_on_curve(t,vals[keys[2]][0],vals[keys[2]][1],ax,color=cs[2])
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
                  # plt.show()
def plot_range_on_curve(t,curve, bounds, ax:plt.axes,color):
      upper = np.add(curve,bounds)
      lower = np.subtract(curve,bounds)
      # x = np.linspace(0,len(upper),len(upper))
      ax.fill_between(t,upper,lower,color=color,alpha=0.2,label='_')
      return ax
