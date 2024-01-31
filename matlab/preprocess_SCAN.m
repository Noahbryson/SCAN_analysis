%% Setup
close all
loadBCI2kTools;
Subject = 'BJH045'; % String of Subject Name
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir. 
DataPath = sprintf("%s/Box/Brunner Lab/DATA/SCAN_Mayo/%s",user,Subject); % Path to data
checkDir(DataPath); % check if data dir exists
% Load Data and Metadata
channels = loadElectrodeChannels(DataPath); % channel discription in parent subject directory, encodes they type and name of each channel
    
dirContents = dir(DataPath);
tgtFile = 'run'; % str for folder to parse in parent subject directory
dataLocs = parseDir(dirContents,tgtFile,'beans');
pathName = dataLocs{1}; % select file wanted if multiple runs of this experiment (ie pre and post ablation), alphabetical order
tDir = sprintf('%s/%s',DataPath,pathName); % path to specific session
files = dir(tDir); % file list
fname = parseDir(files,'dat','_');
fname = fname{end}; % extract fname from cell array
[data,states,parms]=load_bcidat(strcat(tDir,'/',fname),1); % load BCI2000 dat file
secondaryBCIflag = ismember('gUSB',channels.Var6);

if secondaryBCIflag
fname_sub = strsplit(fname,'.');
tgt = fname_sub(1);
tgt = strcat(tgt,'_1.dat');
[data2,states2,parms2] = load_bcidat(strcat(tDir,'/',tgt{1}),1);
[data2,states2] = resampleSecondaryData(data2,states2,parms2.SamplingRate.NumericValue,parms.SamplingRate.NumericValue,0);
[DATA1,DATA2,STATES1,STATES2] = alignSecondaryData(data, data2,states,states2);
[data,states] = aggregateData(DATA1,DATA2,STATES1,STATES2);
end
[keys,type] = labelDataChannels(data,channels); % generate labels from data and channel description
saveDir = strcat(tDir,'/preprocessed'); % path to save dir for preprocessed files
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





function [X,Y] = resampleSecondaryData(data,states, fs, target_fs,plot)
num_samp = size(data,1)/fs*target_fs;
X = zeros(num_samp,size(data,2));
Y = states;
for i=1:size(data,2)
    X(:,i) = resample(data(:,i),target_fs,fs,100);
end
fields = fieldnames(states);
for i=1:length(fields)
    temp = Y.(fields{i});
    temp = cast(temp,'single');
    temp = resample(temp,target_fs,fs,10);
    Y.(fields{i}) = temp;
end
if plot
figure
t1 = linspace(0,1,size(data,1));
t2 = linspace(0,1,num_samp);
subplot(2,2,1)
plot(t1,data(:,1))
subplot(2,2,2)
plot(t2,X(:,1));
subplot(2,2,3)
[Pxx,f] = pwelch(data(:,1),[],[],[],fs);
semilogy(f,Pxx)
subplot(2,2,4)
[Pxx,f] = pwelch(X(:,1),[],[],[],target_fs);
semilogy(f,Pxx)
end
end

function [DATA1,DATA2,STATES1,STATES2] = alignSecondaryData(data1, data2,states1,states2)
sync1 = double(states1.DC04);
sync1 = abs(sync1-mean(sync1))/max((sync1));
sync2 = double(states2.DigitalInput4);
sync2 = abs(sync2-mean(sync2))/max((sync2));

thresh1 = 3*std(sync1);
thresh2 = 3*std(sync2);

% figure(1)
% hold on
% plot(sync1)
% yline(thresh1)
% plot(sync2)
% yline(thresh2)
% legend('sync1','thresh1','sync2','thresh2')
% hold off
x1 = detectThresholdCrossing(sync1, thresh1, 3500);
x2 = detectThresholdCrossing(sync2, thresh2, 3500);
avg_offset = cast(mean(x1-x2),'int32');
if avg_offset >= 0 % primary data lagging secondary data
    DATA1 = zeros(size(data1,1)-avg_offset+1,size(data1,2));
    STATES1 = struct;
    for i=1:size(data1,2)
        DATA1(:,i) = shiftTimeseries(data1(:,i),avg_offset);
    end
    fields = fieldnames(states1);
    for i=1:length(fields)
        temp = shiftTimeseries(states1.(fields{i}),avg_offset);
        STATES1.(fields{i}) = temp;
    end
    DATA2 = zeros(size(data2,1)-avg_offset+1,size(data2,2));
    STATES2 = struct;
    for i=1:size(data2,2)
        DATA2(:,i) = truncateTimeseries(data2(:,i),avg_offset);
    end
    fields = fieldnames(states2);
    for i=1:length(fields)
        temp = truncateTimeseries(states2.(fields{i}),avg_offset);
        STATES2.(fields{i}) = temp;
    end
else % primary data leading secondary data
    DATA2 = zeros(size(data2,1)-avg_offset+1,size(data2,2));
    STATES2 = struct;
    for i=1:size(data2,2)
        DATA2(:,i) = shiftTimeseries(data2(:,i),avg_offset);
    end
    fields = fieldnames(states2);
    for i=1:length(fields)
        temp = shiftTimeseries(states2.(fields{i}),avg_offset);
        STATES2.(fields{i}) = temp;
    end
    DATA1 = zeros(size(data1,1)-avg_offset+1,size(data1,2));
    STATES1 = struct;
    for i=1:size(data1,2)
        DATA1(:,i) = truncateTimeseries(data1(:,i),avg_offset);
    end
    fields = fieldnames(states1);
    for i=1:length(fields)
        temp = truncateTimeseries(states1.(fields{i}),avg_offset);
        STATES1.(fields{i}) = temp;
    end





end
sync1 = double(STATES1.DC04);
sync1 = abs(sync1-mean(sync1))/max((sync1));
sync2 = double(STATES2.DigitalInput4);
sync2 = abs(sync2-mean(sync2))/max((sync2));
figure(2)
hold on
plot(sync1)
yline(thresh1)
plot(sync2)
yline(thresh2)
legend('sync1','thresh1','sync2','thresh2')
hold off
end

function [data, states] = aggregateData(DATA1,DATA2,STATES1,STATES2)
addedDims = size(DATA2,2);
data = zeros(size(DATA1,1),size(DATA1,2)+addedDims);
data(:,1:size(DATA1,2)) = DATA1;
for i=1:addedDims
    idx = size(DATA1,2) + i;
    data(:,idx) = DATA2(:,i);
end
states = STATES1;
end


function X = shiftTimeseries(data, shift)
 X = data(shift:end);
end
function X = truncateTimeseries(data,shift)
X = data(1:end-shift+1);
end
function targets = parseDir(dir,target,exclude)
targets = {};
for i=1:length(dir)
    if (contains(dir(i).name,target)) && (~contains(dir(i).name,exclude))
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