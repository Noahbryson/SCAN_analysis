from pathlib import Path
from tkinter.filedialog import askopenfilename
import os
import shutil

root = Path('/Users/nkb/Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/raw_data/share')
out = root.parent / 'stimcodes'
for s in root.glob('Mayo*'):
      sub = s.name
      os.makedirs(out/sub,exist_ok=True)
      fout = out / sub / 'samplerun.dat'
      if not os.path.exists(fout):
            fin = askopenfilename(initialdir=s)
            shutil.copy(fin,fout)