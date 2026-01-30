%% Seizure Network Figure
BCI2KPath = '/Users/nkb/Documents/NCAN/BCI2000tools';
addpath(genpath('/Users/nkb/Documents/NCAN/code/MATLAB_tools'))
addpath(genpath('/Users/nkb/Documents/NCAN/code/SCAN_analysis/'))
bci2ktools(BCI2KPath);
rootPath = '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo';

%%
subject = "BJH041";
% seizure starts near 48-50s
brainDir = "/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/brain/MNIbrain_destrieux.mat";
ddir = "/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/stimulation_mapping/50Hz";
subsetDir = '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/brain/electrodes_precentral&central.csv';
subsetDir = '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/brain/electrodes_Motor-Strip.csv';
subsetDir = '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/brain/electrodes_Motor-Strip_cmap-clusters.xlsx';

% "K13-14_seizure.dat";
subset_table = readtable(subsetDir);
channel_ROIs = subset_table.electrodes;
files = dir(fullfile(ddir,'*.dat'));
brain = load(brainDir);
[signals, states,param]=load_bcidat(fullfile(files(1).folder,files(1).name));
fs = param.SamplingRate.NumericValue;
cutoff = fs*180;
signals = signals(1:cutoff,:);
f = fieldnames(states);
for i=1:length(f)
    x = states.(f{i});
    states.(f{i}) = x(1:cutoff);
end
channels = param.ChannelNames.Value;
chans = channels(1:194);
rmIdx = contains(chans,'REF');
chans = chans(~rmIdx);
loc = ismember(channels,chans);
chans = cellfun(@(s) replace(s,' ',''),chans,'UniformOutput',false);

% signals = signals(:,~rmIdx);signals = signals(:,1:194);
data = signals(:,loc);
trajs = unique(regexprep(chans, '\d', ''));
res = struct();
count = 1;
for i=1:length(trajs)
locs = contains(chans,trajs{i});
d = data(:,locs);
c = chans(locs);
% nums = cellfun(@(s) str2num(regexp(s, '\d+', 'match')),c,'UniformOutput', false);
nums = cellfun(@(s) str2double(regexp(s, '\d+', 'match')), c, 'UniformOutput', false);
nums = cell2mat(nums);
[nums,sIdx] = sort(nums);
d = d(:,sIdx);
c = c(sIdx);
for j=1:length(c)-1
    sig = d(:,j) - d(:,j+1);
    dec = sprintf('%s%d-b-%d',trajs{i},nums(j),nums(j+1));
    res(count).chan = dec;
    res(count).sig = sig;


    count = count +1;
end
end
yshift = 400;
sigs = getHighPassData(data,2,2,fs);
figure(9)
hold on
ticks = [];
slice = [100:180];
for i=1:length(chans(slice))
    plot(sigs(:,slice(i))+(i-1)*yshift);
    ticks = [ticks yshift*(i-1)];
end
hold off
yticks(ticks)
yticklabels(chans(slice))
chans = {res.chan}';
clear signals data



data = [res.sig];
data = getHighPassData(data,2,2,fs);

%% SZ Network Timeseries one ax
downsamp_factor = 20;
shift_amount = 1250;

stim = states.DC04;
stim = logical(stim-mean(stim));
fs = param.SamplingRate.NumericValue/downsamp_factor;
stim = stim(1:downsamp_factor:end);
dat = data(1:downsamp_factor:end,:);

fig = figure(1);
[~, exclude] = ismember({'JL9-b-10','JL10-b-11'},chans);
if sum(exclude) > 0
indexer = ones(size(chans));
indexer(exclude) = 0;
indexer = indexer > 0;
chans = chans(indexer);
dat = dat(:,indexer);
end
% [~, exclude] = ismember({'JL9-b-10','JL10-b-11'},channel_ROIs);
% indexer = ones(size(channel_ROIs));
% if sum(exclude) > 0
% indexer(exclude) = 0;
% indexer = indexer > 0;
% channel_ROIs = channel_ROIs(indexer);
% 
% end 
if ~isempty(channel_ROIs)
    [matches, chan_subset] = ismember(channel_ROIs,chans);
    chan_subset(chan_subset==0) = [];
    
    if ismember({'R','G','B'},subset_table.Properties.VariableNames)
        [~,ic] = ismember(chans(chan_subset),channel_ROIs);
        subset_colors = table2array(subset_table(:,{'R','G','B'}));
        pallete = subset_colors(ic,:);
    else
        pallete = color_gradient(length(chan_subset),[1 0 0], [0 0 1]);

    end
    
end


% chan_subset = [1:28, 34:115, 118:178];
% chan_subset = [86:92, 130:137, 152:156];

% chan_subset = 1:170;

t = linspace(0,size(dat,1)/fs,size(dat,1));
tcl = tiledlayout(1,1);
colors = repmat([0 0 0],length(chan_subset),1);
colors(4:5,:) = repmat([53, 152, 204]/255,2,1);
colors(13,:) = [1 0 0];
colors(14:15,:) = repmat([105 201 202]/255,2,1);
colors(17:19,:) = repmat([75 96 247]/255,3,1);
ax = gca;

% ax.XTickLabel = [];
ax.XAxis.Visible = 'On';
for i=1:length(chan_subset)
    yshift = i * shift_amount - shift_amount;
    hold on
    plot(t,dat(:,chan_subset(i))+yshift,'Color',pallete(i,:))
    % plot(t,dat(:,chan_subset(i))+yshift,'Color',[0,0,0])
    % plot(t,dat(:,chan_subset(i))+yshift)
    plot(t(stim),dat(stim,chan_subset(i))+yshift,'LineWidth',2,'Color',[1 0 0])
    
end
% legend(chans{chan_subset})
yt = linspace(0,yshift,length(chan_subset));
ytl = chans(chan_subset);
yticks(yt)
yticklabels(ytl)
% ytickangle(45)
hold off
linkaxes(ax,'xy')
xlim([20,130])
% ylim([min(0,yshift)-shift_amount,max(0,yshift)+shift_amount])
xlabel(tcl, 'Time (s)');
ax.Position = [0.1 0.02 0.8 0.95];
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',8)

%%
figpath="/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/Writing/manuscripts/SCAN ABLATION 2025/figures/3.SZ Network";

% exportgraphics(fig,fullfile(figpath,'SZ_ephys.svg'))
% print(fig,fullfile(figpath,'SZ_ephys.svg'),'-dsvg')

exportgraphics(gcf,fullfile(figpath,'SZ_ephys_200Hz.eps'),'ContentType','vector')
% exportgraphics(gcf,fullfile(figpath,'SZ_ephys_200Hz.eps'))


% ylabel(tcl, 'Amplitude');

%% SZ Network Timeseries
% downsamp_factor = 20;
% stim = states.DC04;
% stim = logical(stim-mean(stim));
% fs = param.SamplingRate.NumericValue/downsamp_factor;
% stim = stim(1:downsamp_factor:end);
% dat = data(1:downsamp_factor:end,:);
% 
% fig = figure(1);
% 
% chan_subset = [86:92, 130:137, 152:156];
% offsets = std(data(:,chan_subset));
% t = linspace(0,size(dat,1)/fs,size(dat,1));
% tcl = tiledlayout(length(chan_subset),1);
% colors = repmat([0 0 0],length(chan_subset),1);
% colors(4:5,:) = repmat([53, 152, 204]/255,2,1);
% colors(13,:) = [1 0 0];
% colors(14:15,:) = repmat([105 201 202]/255,2,1);
% colors(17:19,:) = repmat([75 96 247]/255,3,1);
% for i=1:length(chan_subset)
% 
%     ax(i) = nexttile(tcl);
%     if i < length(chan_subset)
%         ax(i).XTickLabel = [];
%         ax(i).XAxis.Visible = 'off'; 
%     end
%     hold on
%     plot(t,dat(:,chan_subset(i)),'Color',colors(i,:))
%     plot(t(stim),dat(stim,chan_subset(i)),'LineWidth',2,'Color',[1 0 0])
%     ylabel(chans{chan_subset(i)})
%     hold off
% end
% 
% linkaxes(ax,'xy')
% xlim([45,120])
% ylim([-3000 3000])
% xlabel(tcl, 'Time (s)');
%%

function pallete = color_gradient(n,color1,color2)

c1 = max(0,min(color1,1));
c2 = max(0,min(color2,1));
r = linspace(c1(1),c2(1),n);
g = linspace(c1(2),c2(2),n);
b = linspace(c1(3),c2(3),n);
pallete = [r' g' b'];
end