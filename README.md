# SCAN_analysis
Repository for analysis of brain signals during motor tasks in patients with sEEG coverage in motor cortex.

## MATLAB
MATLAB functions are used for quick visualization of data, along with preprocessing and parsing data into proper formats.
Formatted data is then exported for bulk analysis in Python. 
- some functions written in MATLAB may also be imported into python using the MATLAB engine. 
### Modules
**preprocess_SCAN.m**
- performs the above preprocessing and dataformatting steps.

## Python 
Python is the core data analysis modality for this code.
Source data should be formatted as the following:
	1. Labeled Data - <Subject Name>.mat
	2. Channel Types - channeltypes.mat
	3. States - states.mat
	4. Stimulus Codes - stimuli.mat
All data should be in the preprocessed folder for each subject within my personal box folder for data. 

### Modules
**SRC**
- TBD