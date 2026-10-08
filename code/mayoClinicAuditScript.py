import os
from pathlib import Path
from src.SCAN_SingleSessionAnalysis import *
import matplotlib.pyplot as plt
import sys
from PyBrain.modules.VERA_PyBrain import PyBrain
from src.functions.graphics import default_gradient


import platform
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
    dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
    dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"

subjects = [i.name for i in dataPath.glob('Mayo_*/')]

gammaRange = [70,170]
brainType = "patient_brain"
laplacian = False
bipolar = True
loadData=False
# loadData=True
save = True
ERP_flag = False
showFlag = False

overwrite_flag = True

side = 'both'

if bipolar:
    reref = 'bipolar'
elif laplacian:
    reref = 'laplacian'
else:
    reref = 'common'



subject = 'Mayo_JBG'
subjects = [subject]


error_log = {}
for subject in sorted(subjects):
    sessions = [i.name for i in Path(dataPath/subject).glob(f'{subject}*/')]
    for session in sessions:
        aggpath = dataPath / 'Aggregate' / f'{subject}_{session}'
        bp = dataPath/subject/'brain'/f'{brainType}.mat'
        
        brainFlag = False
        brain = None
        if not os.path.exists(aggpath/'channel_classifications.csv') or overwrite_flag:
            print(f'Running Analysis on Patient: {subject}\nsession: {session}')
            try:
                if subject == 'BJH041':
                    remove = ['OR']
                else: remove=[]
                a = SCAN_SingleSessionAnalysis(dataPath,subject,session,remove_trajectories=remove,
                    load=loadData,plot_stimuli=False,gammaRange=gammaRange,refType=reref)

                a.plot_session_EMG_motor_onsets()
                if False:
                    plt.show()
                else:
                    a.plot_movement_latencies()
                    
                    a.save_movement_latencies()
                    plt.show(block=False)
                    if ERP_flag:
                        a.run_ERP_processing(plot=True,save=True,show=False)
                    r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(save=save,makePlots=save)
                    sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals,alpha=0.05)
                    effect_of_interest =r_sq
                    effect_name = 'r_sq'
                    datasubset = sig_chans
                    intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(effect_of_interest,[],
                                            save=save,label='significant',thresh=.1,comparison='')
                    channel_classifcation_out = []
                    if brainFlag:
                        electrodemap,regionsLocs = brain._get_ROI_map()
                        for i in channel_classifcation:
                            if i in sig_chans:
                                channel_classifcation_out.append(f'{i},{channel_classifcation[i]},1,{electrodemap[i]}\n')
                            else: 
                                channel_classifcation_out.append(f'{i},{channel_classifcation[i]},0,{electrodemap[i]}\n')
                    with open(aggpath/'channel_classifications.csv','w') as fp:
                        fp.write('channel,class,significant,region\n')
                        fp.writelines(channel_classifcation_out)
                    intereffectors = [i.replace('_','') for i in intereffectors]
                    nonspecifics = [i.replace('_','') for i in nonspecifics]

                    cmap_resolution=1

                    tuning_colors,target_colors = default_gradient(cmap_resolution,run_circular=False)
                    tuning,chan_tuning_colors,angle_key, color_array = a.somatotopic_tuning(r_sq,tuning_colors=tuning_colors,plotCMAP=True)
                    colorLab = list(angle_key.keys())
                    labColor = [i['color'] for i in angle_key.values()]
                    t_ = [[i,j['color']] for i,j in angle_key.items()]
                    t_names = [i[0] for i in t_]
                    t_colors = [i[1] for i in t_]
                    allChans = tuning['channel'].to_list()
                    shared_rep = a.shared_representation(effect_of_interest,sig_chans)


                    print('intereffectors')
                    for i in intereffectors:
                            print(i)
                    print('\n\nnonspecifics')
                    non_specific= ['inter','foot-face','hand-face','hand-foot']
                    multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
                    targets = []
                    for i in (multiMotor):
                        print(i[0],': ',i[1])
                        targets.append(i[0])
                    
            except Exception as err:
                print(err)
                print(f'aborting {subject}')
                error_log[subject] = str(err)
                
plt.show()