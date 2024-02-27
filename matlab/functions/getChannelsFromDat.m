


close all
loadBCI2kTools;
Subject = 'BJH045'; % String of Subject Name
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir. 
DataPath = sprintf("%s/Box/Brunner Lab/DATA/SCAN_Mayo/%s",user,Subject); % Path to data
checkDir(DataPath); % check if data dir exists
% Load Data and Metadata
    
dirContents = dir(DataPath);
tgtFile = 'run'; % str for folder to parse in parent subject directory
dataLocs = parseDir(dirContents,tgtFile,'beans');
% adjust this index for run number
pathName = dataLocs{1}; % select file wanted if multiple runs of this experiment (ie pre and post ablation), alphabetical order
tDir = sprintf('%s/%s',DataPath,pathName); % path to specific session
files = dir(tDir); % file list
fname = parseDir(files,'dat','_');
fname = fname{end}; % extract fname from cell array
fp = fullfile(tDir,fname);
chans = getChannels(fp);





function x=getChannels(path)
[data,states,params] = load_bcidat(path,1);
channels = data;

end