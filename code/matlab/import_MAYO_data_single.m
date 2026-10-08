user = expanduser('~');

paths = struct();
paths.input_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/task_files");
paths.brain_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/imaging");
paths.dump_root = fullfile(user,"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo");
paths.emg_table = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/emg_description.csv");
paths.stimuli_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/raw_data/stimcodes");

exportFlag = false;
subject = 'Mayo_AML';

[subjectLog,subjectPass] = import_MAYO_data(subject,paths,exportFlag,true);
