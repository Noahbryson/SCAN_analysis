from pathlib import Path
import os
from src.SCAN_group_analysis import SCAN_group_analysis
from matplotlib import pyplot as plt




subject_sessions = ['BJH041_pre_ablation','BJH041_post_ablation']
import platform
localEnv = platform.system()
userPath = Path(os.path.expanduser('~'))
if localEnv == 'Windows':
      dataPath = userPath / r"Box\Brunner Lab\DATA\SCAN_Mayo"
else:
      dataPath = userPath/"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"
a = SCAN_group_analysis(dataPath/'Aggregate',subject_sessions)
# a.compare_EMG_isolation(subject_sessions[0],subject_sessions[1])
a.compare_rsq(subject_sessions[0],region_search_strs='central')
a.compare_shared_rep(subject_sessions[0],region_search_strs='central')
# a.LoadERPs()

plt.show()