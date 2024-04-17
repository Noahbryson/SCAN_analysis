%% Setup
close all
addpath('/Users/nkb/Documents/NCAN/code/MATLAB_tools')
BCI2KPath = '/Users/nkb/Documents/NCAN/BCI2000tools';
bci2ktools(BCI2KPath);
Subject = 'BJH041'; % String of Subject Name
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir. 
if ispc
    DataPath = sprintf("%s/Box/Brunner Lab/DATA/SCREENING/%s",user,Subject); % Path to data
else
    DataPath = sprintf("%s/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCREENING/%s",user,Subject); % Path to data
end
checkDir(DataPath); % check if data dir exists
% Load Data and Metadata
channels = loadElectrodeChannels(DataPath); % channel discription in parent subject directory, encodes they type and name of each channel
    
dirContents = dir(DataPath);
dataDirs = {'sensory','motor','sensory-motor'};

dataLocs = parseDir(dirContents,dataDirs,'.csv');


% adjust this index for run number
for i=1:length(dataLocs)
    pathname = dataLocs{i}; % select file wanted if multiple runs of this experiment (ie pre and post ablation), alphabetical order
    
    fprintf('processing of %s starting\n',pathname)
    tDir = sprintf('%s/%s',DataPath,pathname); % path to specific session
    files = dir(tDir); % file list
    fname = parseDir(files,'dat','_');
    fname = fname{end}; % extract fname from cell array
    [data,states,parms]=load_bcidat(strcat(tDir,'/',fname),1); % load BCI2000 dat file
    secondaryBCIflag = ismember('gUSB',channels.Var6);
    fname_sub = strsplit(fname,'.');
    tgt = fname_sub(1);
    tgt = strcat(tgt,'_1.dat');
    if (secondaryBCIflag && pathname ~= "sensory" && isfile(strcat(tDir,'/',tgt{1})))% if there are secondary EMG recordings!!!!!
    
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
    
    test = writeChannelDescriptions(saveDir,keys,type,1); % write channel decriptions as a structure to .mat (v7.0) files
    states = writeStates2MAT(saveDir,states); % write states as a structure to .mat (v7.0) files
    writeStimuliCodes(parms,saveDir) % write stimuli code parm as a structure to .mat (v7.0) files -> will eventually reshape and encode other metadata like sampling rate
    writeMATwithHeader(saveDir,Subject,data,keys,1); % write labeled data as a structure to .mat (v7.0) files
    fprintf('processing of %s finished\n',pathname)
end


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

x1 = detectThresholdCrossing(sync1, thresh1, 10000);
x2 = detectThresholdCrossing(sync2, thresh2, 10000);
avg_offset = cast(mean(x1-x2),'int32'); 
%avg_offset > 0 primary lags secondary, avg_offset < 0 primary leads secondary

% if datastreams are different lengths, adjust them to the length of
% the longer one. if the smaller stream lags the larger, truncate larger stream. if the
% smaller stream leads the larger stream, shift the larger stream. 
stream_diff = size(data1,1) - size(data2,1); %positive: prim > sec, negative prim < sec
if stream_diff > 0 && avg_offset > 0
% prim bigger and lagging
data1 = data1(stream_diff+1:end,:);
elseif stream_diff < 0 && avg_offset > 0
% prim smaller and lagging
data2 = data2(1:end-(abs(stream_diff)),:);
elseif stream_diff > 0 && avg_offset < 0
% prim bigger and leading
data1 = data1(1:end-(stream_diff-1),:);
elseif stream_diff < 0 && avg_offset < 0    
% prim smaller and leading
data2 = data2(stream_diff+1:end,:);
end

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


function [keys,type] = labelDataChannels(data, channels)
N_lab = size(channels,1) - 1;
N_chan = size(data,2);
if N_lab ~= N_chan
    disp('channel and label mismatch. empty channels in secondary bci?\n using channel length as N, be sure to verify correctness');
    N_chan = N_lab;
end
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