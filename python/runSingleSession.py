import os
from pathlib import Path
from SCAN_SingleSessionAnalysis import SCAN_SingleSessionAnalysis

userPath = Path(os.path.expanduser('~'))
dataPath = userPath / "Box\Brunner Lab\DATA\SCAN_Mayo"
subject = 'BJH041'
session = 'pre_ablation'

a = SCAN_SingleSessionAnalysis(dataPath,subject,session,load=True)

