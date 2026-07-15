%% Isolate Electrode Locations for Identified Channels, export to .dat for freeview.
clear 
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir. 
if ispc
    boxpath = fullfile(user,"Box/Brunner Lab"); % Path to data
     BCI2KPath = "C:\BCI2000\BCI2000";
else
    boxpath =  fullfile(user,'Library/CloudStorage/Box-Box/Brunner Lab'); % Path to data
    BCI2KPath = '/Users/nkb/Documents/NCAN/BCI2000tools';
end
datapath = fullfile(boxpath,"/DATA/SCAN_Mayo");
addpath(genpath(fullfile(user,'Documents/NCAN/code/BLAES_stimSweep')));
addpath(genpath(fullfile(user,'Documents/NCAN/code/MATLAB_tools')));
bci2ktools(BCI2KPath);
d = struct;
sessions = dir(fullfile(datapath,'Aggregate/BJH*'));
for i =1:length(sessions)
    fpts = dir(fullfile(sessions(i).folder,sessions(i).name,'*_channel_sort_significant.mat'));
    fp = fullfile(fpts(1).folder,fpts(1).name);
    subject = strsplit(sessions(i).name,'_');
    subject = subject{1};
    elecPath = fullfile(datapath,'electrodes',subject);
    e = load_electrodes(elecPath);
    dat = load(fp);
    d(i).name = sessions(i).name;
    d(i).inter = cellstr(dat.inter);
    targets = parseElectrodes(dat.inter,e);
    d(i).targets = targets;
    fname = fullfile(elecPath,sprintf('%s_intereffectors.dat',subject));
    writeTargetDats(targets,fname);
end




function e = load_electrodes(dirpath)
e = struct;
files = dir(fullfile(dirpath,'*.dat'));
fnames = {files.name};
exclude = 'intereffector';
excludeIdx = ~contains(fnames,exclude,'IgnoreCase',true);
files = files(excludeIdx);
count = 1;
for i =1:length(files)
    name = strsplit(files(i).name,'.');
    name = name{1};
    xx = readlines(fullfile(files(i).folder,files(i).name));
    numpts =  str2num(strjoin(regexp(xx(end-1), '\d+', 'match'),''));
    blankIdx = strlength(xx) == 0;
    blankIndices = find(blankIdx);
    if sum(blankIdx) > 0 && sum(blankIndices <= numpts) > 0
    j = 1+sum(blankIndices <= numpts);
    else 
    j = 1;
    end
    
    for pp =1:numpts
        e(count).name = sprintf('%s_%d',name,pp);
        
        locs = strsplit(xx(j),' ');
        e(count).x = str2num(locs{1});
        e(count).y = str2num(locs{2});
        e(count).z = str2num(locs{3});   
        count = count +1;
        j = j+1;
    end
end
end

function res = parseElectrodes(targets,electrodes)
targets = cellstr(targets);
e = {electrodes.name};
i = ismember(e,targets);
res = electrodes(i);
end

function writeTargetDats(targets,fp)
N = length(targets);
s = cell(N,1);
fid = fopen(fp,'w');
for i=1:N
    s{i} = sprintf('%0.5f %0.5f %0.5f',targets(i).x,targets(i).y,targets(i).z);
    fprintf(fid,'%0.4f %0.4f %0.4f\n',targets(i).x,targets(i).y,targets(i).z);
end
fprintf(fid,'info\n');
fprintf(fid,'numpoints %d\n',N);
fprintf(fid,'useRealRas 1');
fclose(fid);
end