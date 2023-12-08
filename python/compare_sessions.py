import os
from pathlib import Path
from filters import *
from stat_methods import *
import scipy.io as scio
import scipy.signal as sig
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from SCAN_SingleSessionAnalysis import SCAN_SingleSessionAnalysis
class compare_sessions():
    def __init__(self,
                 parentDir:Path or str,
                 subject:str,
                 analysis_types:str or list = ['rsq','emg','seshEMG'],
                 session_labels: list = ['pre_ablation','post_ablation']
    ):
        if type(parentDir)==str:
            parentDir = Path(parentDir)
        if type(analysis_types)==str:
            analysis_types = [analysis_types]
        self.compareLib = {'rsq':self._rsq_compare, 'emg':self._EMG_compare, 'seshEMG':self._sesh_EMG}
        self.dir = parentDir / subject
        self.exportDir = self.dir/'comparisons'
        self.a_types = analysis_types
        self.subject = subject
        self.sesh_labs = session_labels 
        self.data = self._loadFiles()
        self.annotate = False
        self.override = False

    def _loadFiles(self):
        data = {}
        for conf in self.a_types:
            df = pd.DataFrame()
            for lab in self.sesh_labs:
                loc = self.dir/lab/'analyzed'
                for f in os.listdir(loc):
                    if f.lower().find(conf) >-1 and f.lower().find('full')<0:
                        name = f.split('.')[0]
                        d = scio.loadmat(loc/f,simplify_cells=True)
                        dat = [[k,v] for k,v in d.items() if k[0]!='_']
                        df_in = [[name,lab]+i for i in dat]
                        cols = ['task','session','channel',conf]
                        temp = pd.DataFrame(df_in,columns=cols)
                        df = pd.concat([df,temp],ignore_index=True)
                full = scio.loadmat(loc/'fullEMG.mat',simplify_cells=True)
                data[f'{lab}_full_emg'] = {k:v for k,v in full.items() if k[0]!='_'}
            data[conf] = df
        return data

    def _rsq_compare(self,export):
        data = self.data['rsq']
        result = pd.DataFrame()
        pre = data.query("session=='pre_ablation'").copy()
        pre.reset_index(inplace=True)
        post = data.query("session=='post_ablation'").copy()
        post.reset_index(inplace=True)
        assert (pre[['task','channel']] == post[['task','channel']]).all().all(), 'channels are not aligned'
        result['channel'] = pre['channel']
        result['task'] = pre['task']
        result['delta_rsq'] = post['rsq'] - pre['rsq']
        tasks = set((result['task'].to_list()))
        output = {}
        for task in tasks:
            t = result.query('task==@task')
            output[task[2:]] = {i:j for i,j in zip(t['channel'].to_list(),t['delta_rsq'].to_list())}
        # output = {i:result[i].to_list() for i in result.columns}
        if export:
            for task,d in output.items():
                scio.savemat(self.exportDir /f'rsq_compare_{task}.mat',d)
        print('Finished r squared analysis')
    def _EMG_compare(self,export):
        return
        muscleMapping = {'emg_1_Hand':['wristExtensor', 'ulnar'], 'emg_3_Foot':['TBA'],'emg_2_Tongue':['tongue']}
        result = pd.DataFrame()
        data = self.data['emg']
        pre = data.query("session=='pre_ablation'").copy()
        pre.reset_index(inplace=True)
        post = data.query("session=='post_ablation'").copy()
        post.reset_index(inplace=True)
        tasks = set(data['task'].to_list())
        # muscles = set(data['channel'].to_list())
        for task in tasks:
            for m in muscleMapping[task]:
                pre_m =  pre.query( 'task==@task & channel==@m')
                post_m = post.query('task==@task & channel==@m')
                prm = pre_m['emg'].values[0]
                pom = post_m['emg'].values[0]
                for e in range(10):
                    fig, (ax, axx,axm) = plt.subplots(3,1,sharex=True)
                    fig.suptitle(f'{task} {m}')
                    
                    a = prm[e]
                    b = pom[e]
                    r = [len(a),len(b)]
                    r = min(r)
                    a = a[0:r]
                    b = b[0:r]
                    thresh = 0.5*max(a)
                    a_peaks = sig.find_peaks(a,height=thresh,distance=1000)
                    thresh = 0.5 *max(b)
                    b_peaks= sig.find_peaks(b,height=thresh,distance=1000)
                    plot_peakTraces(ax,a,a_peaks)
                    plot_peakTraces(axx,b,b_peaks)
                    axm.plot(a/max(abs(a))-b/max(abs(b)))
                    plt.show()
                    # breakpoint()
    def annotate_session_EMG(self,export,mus,muscleMapping,pre_stimuli,post_stimuli,pre,post):
        output = {}
        df = pd.DataFrame()
        for iii, m in enumerate(mus):
            a_codes = pre_stimuli[muscleMapping[m]]
            b_codes = post_stimuli[muscleMapping[m]]
            a=pre[m];b=post[m]
            a_points = []
            t=np.linspace(0,len(a)/2000,len(a))
            for idx,(i,e) in enumerate(a_codes):
                i = i-500
                e = e +500
                fig, ax = plt.subplots(1,1,num = f'{m} pre')
                ax.set_title(f'pre {m} {idx+1 + (10*iii)}/80')
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                a_points.extend(self._annotate_trial(t,a[i:e],[i,e],ax,i,e))
                plt.close()
            b_points = []
            for idx,(i,e) in enumerate(b_codes):
                idx = idx + 10
                i = i-500
                e = e +500
                # t=np.linspace(0,(e-i)/2000,(e-i))
                fig, ax = plt.subplots(1,1,num = f'{m} pre {idx + 1 + (10*iii)}')
                ax.set_title(f'post {m} {idx+1 + (10*iii)}/80')
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                b_points.extend(self._annotate_trial(t,b[i:e],[i,e],ax,i,e))
                plt.close()
            a_on = [i[0] for i in a_points[0::2]]
            a_off = [i[0] for i in a_points[1::2]]
            b_on = [i[0] for i in b_points[0::2]]
            b_off = [i[0] for i in b_points[1::2]]
            a_dur = []
            a_num = len(a_on)
            for i,j in zip(a_on,a_off):
                a_dur.append(j-i)
            b_dur = []
            b_num = len(b_on)
            for i,j in zip(b_on,b_off):
                b_dur.append(j-i)        

            dat = [m,[a_dur],[a_num],[b_dur],[b_num],[a_on],[a_off],[b_on],[b_off]]
            cols = ['m','pre duration','pre num','post duration','post num','pre start','pre end','post start','post end']
            output[m] = {k:v for k,v in zip(cols,dat)}
            test = pd.DataFrame.from_dict({k:v for k,v in zip(cols,dat)})
            test.set_index('m',inplace=True)
            df = pd.concat([df,test])
        if export:
            df.to_csv(self.exportDir/'ROM_compare.csv')
        return df
    def _annotate_trial(self,t,a,a_codes,ax,i,e):
        plot_peakTraces(ax,t,a,a_codes,True,i,e)
        clicked_pts = plt.ginput(n=-1,timeout=0)
        if len(clicked_pts) >= 2 and len(clicked_pts) % 2 == 0:
            return clicked_pts
        elif len(clicked_pts) % 2 == 1:
            print('odd number of points clicked')
            return self._annotate_trial(t,a,a_codes,ax,i,e)
        else:
            print('min number of points not clicked')
    def _sesh_EMG(self,export):
        print('begin')
        pre = self.data['pre_ablation_full_emg']

        muscleMapping = {'wristExtensor':'1_Hand', 'ulnar':'1_Hand', 'TBA':'3_Foot','tongue':'2_Tongue'}

        pre_stimuli = scio.loadmat(self.dir/'pre_ablation/analyzed/stimuli.mat') 
        pre_stimuli = remove_loading_bs_mat(pre_stimuli)
        pre_State = scio.loadmat(self.dir/'pre_ablation/analyzed/stimcode.mat')
        pre_State = remove_loading_bs_mat(pre_State)
        post = self.data['post_ablation_full_emg']
        post_stimuli = scio.loadmat(self.dir/'post_ablation/analyzed/stimuli.mat')
        post_stimuli = remove_loading_bs_mat(post_stimuli)
        pos_State = scio.loadmat(self.dir/'post_ablation/analyzed/stimcode.mat')
        pos_State = remove_loading_bs_mat(pos_State)
        mus = list(pre.keys())
        if self.annotate:
            romPath = self.exportDir/'ROM_compare.csv'
            if os.path.exists(romPath) and self.override == False:
                annotations = self.loadMovements()
            else:
                annotations = self.annotate_session_EMG(export,mus,muscleMapping,pre_stimuli,post_stimuli,pre,post)
        fig, axs = plt.subplots(4,2,sharex=True,figsize=(10,10))
        for idx,m in enumerate(mus):
            ax =  axs[idx][0]
            axx = axs[idx][1]
            a_codes = pre_stimuli[muscleMapping[m]]
            b_codes = post_stimuli[muscleMapping[m]]
            a=pre[m];b=post[m]
            t=np.linspace(0,len(a)/2000,len(a))
            # thresh = 0.5*max(a)
            # a_peaks = sig.find_peaks(a,height=thresh,distance=1000)
            # thresh = 0.5 *max(b)
            # b_peaks= sig.find_peaks(b,height=thresh,distance=1000)
            plot_peakTraces(ax,t,a,a_codes,True)
            # ax.set_xlabel('time (s)')
            ax.set_ylabel('z-score')
            # ax.legend()
            ax.set_title(f'{m} pre-ablation')
            plot_peakTraces(axx,t,b,b_codes,True)
            # axx.set_xlabel('time (s)')
            axx.set_ylabel('z-score')
            # axx.legend()
            axx.set_title(f'{m} post-ablation')
        fig.suptitle('Movements During Task Pre and Post Ablation')
        fig.subplots_adjust(wspace = 0.075, hspace = 0.2,top=.92,bottom=.05,right=.96,left=.04)
        ax.set_xlabel('time (s)')
        axx.set_xlabel('time (s)')
        
        # for a in axs.flat:
            # a.label_outer()
            # axm.plot(a/max(abs(a))-b/max(abs(b)))
        plt.show()

        print('Finished EMG analysis')
    def analyze_conditions(self,export:bool,plot:bool):
        if export:
            self._verify_export_dir()
        for i in self.a_types:
            self.compareLib[i](export)
    
    def _verify_export_dir(self):
        if not os.path.exists(self.exportDir):
            os.mkdir(self.exportDir)
    def _loadMovements(self):
        movements = pd.read_csv(self.exportDir/'ROM_compare.csv')
        return movements
    def analyzeMovements(self):
        import ast
        movements = self._loadMovements()
        muscles = movements['m'].to_list()
        data = movements[['pre duration','post duration']]
        data = data.applymap(ast.literal_eval)
        data['m'] = movements['m']
        data.set_index('m',inplace=True)
        fig, ax = plt.subplots(1,len(muscles),num='move stats',sharey=True)
        for i,m in enumerate(muscles):
            cols = ['condition','duration']
            target = data.loc[m]
            target = long2wideDF(target,'pre duration','post duration',cols)
            pre = data.loc[m,'pre duration']
            print(m,' ',target['condition'].value_counts())
            post = data.loc[m,'post duration']
            y = [np.mean(pre), np.mean(post)]
            x = ['pre duration','post duration']
            avg = pd.DataFrame(zip(x,y),columns = cols)
            # sns.swarmplot(target,x='condition',y='duration',ax=ax[i],size=3,c=(0,0,0))
            sns.violinplot(target,x='condition',y='duration',ax=ax[i])

            # sns.swarmplot(avg,x='condition',y='duration',ax=ax[i],size=5, c = (1,0,0))
            res,p = stat.ranksums(pre,post)
            ax[i].set_title(f'{m}\nrank: {res}\np: {p}')
            ax[i].set_ylabel('Movement Duration (s)')
        plt.show()
        return 0




def long2wideDF(df,key1,key2,cols):
    a = df[key1]
    a_l = [key1 for _ in range(len(a))]
    b = df[key2]
    b_l = [key2 for _ in range(len(b))]
    a_l.extend(b_l); a.extend(b)
    
    df = pd.DataFrame(zip(a_l,a),columns=cols)
    return df
def plot_peakTraces(ax,x,a,peaks,go=False,i=False,e=False):
    if i and e:
        ax.plot(x[i:e],a,label='_')
    else:
        ax.plot(x,a,label='_')

    if go:
        lab = True
        for p in peaks:
            if type(p) == np.ndarray or type(p) == list:
                if len(p)>1:
                    if lab:
                        ax.axvline(x[p[0]],c=(1,0,0),label='onset')
                        ax.axvline(x[p[1]],c=(0,0,0),label='offset')
                        lab = False
                    else:
                        ax.axvline(x[p[0]],c=(1,0,0),label='_')
                        ax.axvline(x[p[1]],c=(0,0,0),label='_')
            else:
                ax.axvline(x[p])
def remove_loading_bs_mat(dat):
    return {k:v for k,v in dat.items() if k[0]!='_'}







if __name__ == '__main__':
    userPath = Path(os.path.expanduser('~'))
    dataPath = userPath / "Box\Brunner Lab\DATA\SCAN_Mayo"
    subject = 'BJH041'
    a = compare_sessions(dataPath,subject=subject)
    a.annotate = True
    a.override = False
    a.analyzeMovements()
    a.analyze_conditions(export = True,plot=True)