%% Setup
loadBCI2kTools;
Subject = 'BJH041'; % String of Subject Name
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir. 
DataPath = sprintf("%s/Box/Brunner Lab/DATA/SCAN_Mayo/%s",user,Subject); % Path to data
checkDir(DataPath); % check if data dir exists
% Load Data and Metadata
channels = loadElectrodeChannels(DataPath); % channel discription in parent subject directory, encodes they type and name of each channel
dirContents = dir(DataPath);
tgtFile = 'ablation'; % str for folder to parse in parent subject directory
dataLocs = parseDir(dirContents,tgtFile);
pathName = dataLocs{1}; % select file wanted if multiple runs of this experiment (ie pre and post ablation), alphabetical order
tDir = sprintf('%s/%s',DataPath,pathName); % path to specific session
files = dir(tDir); % file list
fname = parseDir(files,'dat');
fname = fname{end}; % extract fname from cell array
[data,states,parms]=load_bcidat(strcat(tDir,'/',fname),1); % load BCI2000 dat file
[keys,type] = labelDataChannels(data,channels); % generate labels from data and channel description
saveDir = strcat(tDir,'/preprocessed') % path to save dir for preprocessed files
if ~exist(saveDir,'dir')
    mkdir(saveDir);
end
writeMATwithHeader(saveDir,Subject,data,keys,1); % write labeled data as a structure to .mat (v7.0) files
test = writeChannelDescriptions(saveDir,keys,type,1); % write channel decriptions as a structure to .mat (v7.0) files
states = writeStates2MAT(saveDir,states); % write states as a structure to .mat (v7.0) files
writeStimuliCodes(parms,saveDir) % write stimuli code parm as a structure to .mat (v7.0) files -> will eventually reshape and encode other metadata like sampling rate

function channels = loadElectrodeChannels(dir)
    fname = sprintf("%s/channels.csv",dir);
    otps = detectImportOptions(fname);
    channels = readtable(fname,otps);
end
function targets = parseDir(dir,target)
targets = {};
for i=1:length(dir)
    if strfind(dir(i).name,target)
        targets{end+1} = dir(i).name;
    end
end
end
function checkDir(dir)

if ~isfolder(dir)
    disp('No Directory Found');
else
    disp('Directory Found')
end
end
function [keys,type] = labelDataChannels(data, channels)
N_chan = size(data, 2);
keys = {N_chan};
type = keys;
channelFlag = 0;
for i=1:N_chan
    ChanLab = channels(i+1,2);
    t = channels(i+1,3);
    t = table2array(t);
    t = t{1};
    if strcmp (t,'') && ~channelFlag
        disp('no channel description')
        channelFlag =1;
    end
    k = table2array(ChanLab);
    k = k{1};
    k = sprintf('%s_%d',k,i);   
    keys{i} = k;
    type{i} = t;
end
end
function signals = writeMATwithHeader(fpath,fname,data,header,removeEmpty)
fpath = strcat(fpath,'/',fname,'.mat');
signals = arrays2Structure(data,header);
if removeEmpty
    vars = fieldnames(signals);
    rm_fields = contains(vars,'EMPTY');
    rm_fields = vars(rm_fields);
    signals = rmfield(signals,rm_fields);
    
end
save(fpath,'signals','-mat','-v7');
end
function channels = writeChannelDescriptions(fpath,channels,chan_types,removeEmpty)
fpath = strcat(fpath,'/channeltypes','.mat');
chan_types = arrays2Structure(chan_types,channels);
if removeEmpty
    vars = fieldnames(chan_types);
    rm_fields = contains(vars,'EMPTY');
    rm_fields = vars(rm_fields);
    chan_types = rmfield(chan_types,rm_fields);
end
save(fpath,'chan_types','-mat','-v7');
end
function states = writeStates2MAT(fpath,states)
fpath = strcat(fpath,'/states','.mat');
save(fpath,'states','-mat','-v7');
end
function writeStimuliCodes(parms,fpath)
stimuli = parms.Stimuli.Value;
stim_codes= cell2table(stimuli(1:end,:));
fpath= strcat(fpath,'/stimuli.mat');
stim_codes = table2struct(stim_codes);
save(fpath,'stim_codes','-mat','-v7');
end
function out = arrays2Structure(data,labels)
labels = makeValidFieldNames(labels);
for j=1:size(data,2)
    out.(labels{j}) = data(:,j);
end
end
function labels = makeValidFieldNames(labels)
for i=1:length(labels)
    labels{i} = matlab.lang.makeValidName(labels{i});
end
end
function table = writeCSVwithHeader(fpath,fname,data,header,removeEmpty)
fpath = strcat(fpath,'/',fname,'.csv');
dataTable = array2table(data);
dataTable.Properties.VariableNames = header;
if removeEmpty
    vars = dataTable.Properties.VariableNames;
    rm_cols = contains(vars,'EMPTY');
    dataTable(:,rm_cols) = [];
end
writetable(dataTable, fpath, 'WriteVariableNames', true);
end