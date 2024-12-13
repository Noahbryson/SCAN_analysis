import os
from pathlib import Path
from src.SCAN_SingleSessionAnalysis import *
import matplotlib.pyplot as plt
import sys
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain


if __name__ == '__main__':
    import platform
    localEnv = platform.system()
    userPath = Path(os.path.expanduser('~'))
    if localEnv == 'Windows':
        dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
    else:
        dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"
    subject = 'BJH041'
    gammaRange = [70,170]
    session = 'aggregate'
    bp = dataPath/subject/'brain'/'brain_MNI.mat'
    bp2 = dataPath/'BJH058'/'brain'/'brain_new.mat'
    brain2= PyBrain(bp2)
    # brain1 = PyBrain(bp)
    # ax = brain1._generateAxis()
    # brain1._plotBrainVolume(ax,opacity=.5,side='both')
    # brain1._plot_electrodes_on_volume(ax,side='both',size=10)
    # brain1._show(ax)
    
    # ax1 = brain1._generateAxis()
    # brain1._plotBrainVolume(ax1,opacity=0.1,color=[250,250, 250])
    # brain1._show(ax1)

    try:
        brain = PyBrain(bp)
    except:
        brain = None
        print('brain path not valid')
    electrodemap,regionsLocs = brain._get_ROI_map()
    x = False
    if x:
        a = SCAN_SingleSessionAnalysis(dataPath,subject,session,remove_trajectories=[],
                load=True,plot_stimuli=False,gammaRange=gammaRange,refType='common')
        fig = plt.figure(figsize=(20,8))
        r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(saveMAT=True)
        sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals,alpha=0.05)
        effect_of_interest = r_sq
        effect_name = 'r_sq'
        datasubset = sig_chans
        datasubset = []
        pp = False
        save_Responses = True
        if pp:            
            a.parse_results_for_triple_responders(effect_of_interest,sig_chans,save=save_Responses,label='significant',thresh=0.1)
            a.parse_results_for_triple_responders(effect_of_interest,save=save_Responses,label='all',thresh=0.1)

