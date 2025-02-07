from pathlib import Path
import os
from src.SCAN_group_analysis import SCAN_group_analysis
from matplotlib import pyplot as plt




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
      # a.LoadERPs()

      plt.show()