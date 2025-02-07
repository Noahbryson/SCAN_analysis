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
    subject = 'BJH071'
    gammaRange = [70,170]
    session = 'aggregate_L'
    brainType = "brain_destrieux"
    laplacian = False
    bipolar = True
    loadData=True
    save = True
    side = 'both'

    if bipolar:
        reref = 'bipolar'
    elif laplacian:
        reref = 'laplacian'
    else:
        reref = 'common'

    bp = dataPath/subject/'brain'/f'{brainType}.mat'
    # brain = PyBrain(bp,subject=subject,brainName=brainType)
    # brain._makeBipolarElectrodes()
    try: #Generation of brain if the file exists, if not data will be parsed as is
        brain = PyBrain(bp,subject=subject,brainName=brainType)
        print(f'\nLoaded {brainType} for {subject} from {bp}\n')
        if bipolar:
                brain._makeBipolarElectrodes()
        if laplacian:
            pass    
        electrodemap,regionsLocs = brain._get_ROI_map()
        centralSulcusInfo = [[i,brain.regionColors[j]] for j,i in enumerate(brain.regions) if i.lower().find('central')>-1]
        brainFlag = True
    except Exception as err:
        brainFlag = False
        print(err)
        brain = None
        print('brain path not valid')




    a = SCAN_SingleSessionAnalysis(dataPath,subject,session,remove_trajectories=[],
        load=loadData,plot_stimuli=False,gammaRange=gammaRange,refType=reref)
    r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(saveMAT=save)
    sig_chans, nonsig_chans, channel_descriptions = a.returnSignificantLocations(p_vals,alpha=0.05)
    effect_of_interest =r_sq
    effect_name = 'r_sq'
    datasubset = sig_chans
    intereffectors, channel_classifcation, nonspecifics = a.parse_results_for_triple_responders(effect_of_interest,datasubset,
                            save=save,label='significant',thresh=.1,comparison='')

    intereffectors = [i.replace('_','') for i in intereffectors]
    nonspecifics = [i.replace('_','') for i in nonspecifics]

        
        
        
    if brainFlag:
        v1=brain._generateAxis(1)
        v1=brain._plotBrainVolume(v1,0.05,[1,1,1],side=side)
        v1=brain._plotBrainRegions(v1,regions=[i[0] for i in centralSulcusInfo], colors=[i[-1] for i in centralSulcusInfo],opacity=.2,side=side)
        
        nonspecifics_locs = [electrodemap[i] for i in nonspecifics]
        intereffector_locs = [electrodemap[i] for i in intereffectors]
        print(channel_classifcation)
        print('intereffectors')
        for i,j in zip(intereffectors,intereffector_locs):
                print(i,': ',j)
        print('\n\nnonspecifics')
        non_specific= ['inter','foot-face','hand-face','hand-foot']
        multiMotor = [[i,j] for i,j in channel_classifcation.items() if j in non_specific]
        targets = []
        for i,j in zip(multiMotor,nonspecifics_locs):
                # if j.find('precentral')>-1 or j.find('S-central')>-1:
                if j.find('central')>-1:
                    print(i[0],': ',j,' ',i[1])
                    targets.append(i[0])
        
        
        # targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
        # targetIdx.extend( brain._getElectrodeIndexFromLabels(labels=nonspecifics))
        targetIdx = brain._getElectrodeIndexFromLabels(labels=targets)
        v1=brain._plot_electrodes_on_volume(v1,electrodeSubset=targetIdx,side=side,color=(0,1,1))
        targetIdx = brain._getElectrodeIndexFromLabels(labels=intereffectors)
        v1=brain._plot_electrodes_on_volume(v1,electrodeSubset=targetIdx,side=side,color=(1,1,0))
        v1.show()
    else:
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
        
