loadBCI2kTools;
Subject = 'BJH041'; %String of Subject Name
user = expanduser('~');
DataPath = sprintf("%s/Box/Brunner Lab/DATA/SCAN_Mayo/%s",user,Subject);
checkDir(DataPath);
channels = loadElectrodeChannels(DataPath);
dirContents = dir(DataPath);
tgtFile = 'ablation';
dataLocs = parseDir(dirContents,tgtFile);
pathName = dataLocs{1};
tDir = sprintf('%s/%s',DataPath,pathName);
files = dir(tDir);
fname = parseDir(files,'dat');
fname = fname{end};
[data,states,parms]=load_bcidat(strcat(tDir,'/',fname),1);
[keys,type] = labelDataChannels(data,channels);
% writeCSVwithHeader(tDir,Subject,data,keys,1)
saveDir = strcat(tDir,'/preprocessed');
if ~exist(saveDir)
    mkdir(saveDir);
end
writeMATwithHeader(saveDir,Subject,data,keys,1);
test = writeChannelDescriptions(saveDir,keys,type,1);
states = writeStates2MAT(saveDir,states);
writeStimuliCodes(parms,saveDir)

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

function table = writeCSVwithHeader(fpath,fname,data,header,removeEmpty)
fpath = strcat(fpath,'/',fname,'.csv');
% headerTable = cell2table(header, 'VariableNames',header);
dataTable = array2table(data);
dataTable.Properties.VariableNames = header;
if removeEmpty
    vars = dataTable.Properties.VariableNames;
    rm_cols = contains(vars,'EMPTY');
    dataTable(:,rm_cols) = [];
end
writetable(dataTable, fpath, 'WriteVariableNames', true);

end

function table = writeMATwithHeader(fpath,fname,data,header,removeEmpty)
fpath = strcat(fpath,'/',fname,'.mat');
% headerTable = cell2table(header, 'VariableNames',header);
table = array2table(data);
table.Properties.VariableNames = header;
if removeEmpty
    vars = table.Properties.VariableNames;
    rm_cols = contains(vars,'EMPTY');
    table(:,rm_cols) = [];
end
save(fpath,'table');
end

function table = writeChannelDescriptions(fpath,channels,chan_types,removeEmpty)
fpath = strcat(fpath,'/channeltypes','.mat');
table = cell2table(chan_types);
table.Properties.VariableNames = channels;
if removeEmpty
    vars = table.Properties.VariableNames;
    rm_cols = contains(vars,'EMPTY');
    table(:,rm_cols) = [];
end
save(fpath,'table');
end

function table = writeStates2MAT(fpath,states)
table = struct2table(states);
fpath = strcat(fpath,'/states','.mat');
save(fpath,"table");
end

function writeStimuliCodes(parms,fpath)
stimuli = parms.Stimuli.Value;
table= cell2table(stimuli(1:end,:));
fpath= strcat(fpath,'/stimuli.mat');
save(fpath,'table')
end

