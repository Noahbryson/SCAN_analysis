%% Aggregate Runs.m
% Script to take multiple runs under the same condition and aggregate them
% together to increase the number of trials in the analysis. 

close all
clear 
loadBCI2kTools;
Subject = 'SLCH020'; % String of Subject Name
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir. 
DataPath = sprintf("%s/Box/Brunner Lab/DATA/SCAN_Mayo/%s",user,Subject); % Path to data
checkDir(DataPath); % check if data dir exists
agg_dir = sprintf('%s/aggregate',DataPath);
save_dir =sprintf('%s/aggregate/preprocessed',DataPath);
if ~exist(agg_dir,'dir')
    mkdir(agg_dir);
    mkdir(save_dir);
end
%% Load Data and Metadata
    
dirContents = dir(DataPath);
tgtFile = 'run'; % str for folder to parse in parent subject directory
dataLocs = parseDir(dirContents,tgtFile,'beans');
runs = struct;
% create a structure with all runs for aggregation. Only states and signals
% time series must be aggregated. 
agg_signals = struct;
states = struct;
for i=1:length(dataLocs) 
    fp = sprintf('%s/%s/preprocessed',DataPath,dataLocs{i});
    temp = load(sprintf('%s/%s.mat',fp,Subject));
    f = fieldnames(temp.signals);
    for j=1:length(f)
        name = f{j};
        if i==1
        agg_signals.(name) = temp.signals.(name);
        else
        agg_signals.(name) = [agg_signals.(name); temp.signals.(name)];
        end

    end
    temp = load(sprintf('%s/states.mat',fp));
    f = fieldnames(temp.states);
    for j=1:length(f)
        name = f{j};
        if i==1
        states.(name) = temp.states.(name);
        else
        states.(name) = [states.(name); temp.states.(name)];
        end

    end
end

%% Save Block
chan_types = load(sprintf('%s/%s/preprocessed/channeltypes.mat',DataPath,dataLocs{1}));
stim_codes = load(sprintf('%s/%s/preprocessed/stimuli.mat',DataPath,dataLocs{1}));
fp = sprintf('%s/%s.mat',save_dir,Subject);
save(fp,'agg_signals','-mat','-v7.3');
fp = sprintf('%s/states.mat',save_dir);
save(fp,'states','-mat','-v7');
fp = sprintf('%s/channeltypes.mat',save_dir);
chan_types = chan_types.chan_types;
save(fp,'chan_types','-mat','-v7');
fp = sprintf('%s/stimuli.mat',save_dir);
stim_codes = stim_codes.stim_codes;
save(fp,'stim_codes','-mat','-v7');
