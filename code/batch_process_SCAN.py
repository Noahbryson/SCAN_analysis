import os
from src.SCAN_SingleSessionAnalysis import SCAN_SingleSessionAnalysis
from pathlib import Path
import json
from platform import system

userpath = Path(os.path.expanduser('~'))
if system() == 'Windows':
      boxpath = userpath
else:
      boxpath = userpath / "Library/CloudStorage/Box-Box"

dataroot = boxpath/ 'Brunner Lab'/'DATA'/'SCAN_Mayo'
with open(dataroot/'subjects.json','r') as fp:
      subjects_info:dict = json.load(fp) 

data_save_root = dataroot/'Aggregate'/'task_power'
latency_save_root = dataroot/'Aggregate'/'movement_latencies'

effect_label = 'r-sq'
# effect_label = 'roc'
overwrite = False
overwrite = True

for subject,session in zip(subjects_info['subjects'],subjects_info['sessions']):
      print(f'-----------------------\n')
      print(f'running export on {subject}: {session}')
      
      data_out = data_save_root/f'{subject}_{session}'
      if not os.path.exists(data_out/f'{effect_label}_{subject}_metrics.json') or overwrite:
            os.makedirs(data_out,exist_ok=True)
            try:
                  a = SCAN_SingleSessionAnalysis(path=dataroot,subject=subject,sessionID=session,load=True,plot_stimuli=False,gammaRange=[70,170],refType='bipolar')
                  # r_sq, p_vals, U_res, d_res,roc_res = a.task_power_analysis(save=True,makePlots=False)
                  # if effect_label == 'r-sq': metric = r_sq
                  # if effect_label == 'd': metric = d_res
                  # if effect_label == 'roc': metric = roc_res
                  # a.export_task_power(effect_label,metric,p_vals,data_out)
                  a.save_movement_latencies(savepath=latency_save_root)
            except KeyError as e:
                  print(e)
                  print('going to next subject')
      else:
            print('export not run')
      print(f'\n\n\n-----------------------')
