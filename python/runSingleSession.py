import os
from pathlib import Path
from src.SCAN_SingleSessionAnalysis import SCAN_SingleSessionAnalysis


if __name__ == '__main__':
    userPath = Path(os.path.expanduser('~'))
    dataPath = userPath / "Box\Brunner Lab\DATA\SCAN_Mayo"
    subject = 'BJH041'
    session = 'post_ablation'

    a = SCAN_SingleSessionAnalysis(dataPath,subject,session,load=True,plot_stimuli=False)
    a.export_epochs(signalType='EMG',fname='emg')
    a.taskPowerCorrelation_analysis(saveMAT=True)
