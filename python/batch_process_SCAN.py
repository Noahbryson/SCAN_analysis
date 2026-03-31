import os
from platform import system
from src import SCAN_SingleSessionAnalysis
from pathlib import Path
import json


userpath = Path(os.path.expanduser('~'))
if system() == 'Windows':
      boxpath = userpath
else:
      boxpath = userpath

dataroot = boxpath/ 'Brunner Lab'/'DATA'/'SCAN_Mayo'
with open(dataroot/'subjects.json','r') as fp:
      subjects_info:dict = json.load(fp) 
for subject,session in subjects_info.items():
      fp = dataroot/subject/session