function [outputArg1,outputArg2] = plot_electrode_labels(brain,labels,electrodeColors,cmap,varargin)
%UNTITLED2 Summary of this function goes here
%   plot the loaded electrode labels onto the brain surface.

%% varagin handling
p = inputParser;
azimuth = 180;
elevation = 0;
titleFlag = 0;
opacity = 100; % percent opacity, set to
forVideo = 0;
full = 1;
SCANonly = 0;
electrodeColors=[62,108,179; 27,196,225; 129,199,238;44,184,149;0,129,145;193,189,47;200,200,200]/256;
conditionIdx = 1;
addOptional(p,'conditionIdx',conditionIdx);
addOptional(p,'forVideo',forVideo);
addOptional(p,'full',full);
addOptional(p,'opacity',opacity)
addOptional(p, 'azimuth', azimuth);
addOptional(p, 'elevation', elevation);
addOptional(p,'titleFlag',titleFlag);
addOptional(p,'electrodeColors',electrodeColors);
addOptional(p,'SCANonly',SCANonly);
parse(p,varargin{:})
SCANonly = p.Results.SCANonly;
azimuth = p.Results.azimuth;
elevation = p.Results.elevation;
titleFlag = p.Results.titleFlag;
opacity = p.Results.opacity;
forVideo = p.Results.forVideo;
electrodeColors = p.Results.electrodeColors;
full = p.Results.full;
conditionIdx = p.Results.conditionIdx;
faceAlpha = opacity/100;

%% make brain
fig = figure;
if full
    fig.WindowState = 'maximized';
end
[annotation_remap,~,name,name_id] = createColormapFromAnnotations(brain.annotation);
surf = plot3DModel(gca,brain.cortex,brain.annotation.Annotation,"FaceVertexCData",cmap,"FaceColor",'interp',"FaceAlpha",faceAlpha);
electrodes = brain.tala.electrodes;
hold on
for i=1:size(labels.vera_idx,1)
    contact = electrodes(labels.vera_idx(i),:);
    cluster_idx = labels.channelClass(i)+1;
    % if labels.SCANflag(i) == 1
    %     contact_size = 3;
    % elseif labels.significant(i) == 0
    %     contact_size = 0.5;
    if labels.significant(i) == 0
        contact_size = 0.5;
    elseif labels.SCANflag(i) == 1
        contact_size = 3;
    else
        contact_size = 0.75;
    end
    if (SCANonly && labels.SCANflag(i)==1)
        plotBallsOnVolume(gca, contact,electrodeColors(5,:),contact_size);
    end
    if ~SCANonly
        plotBallsOnVolume(gca, contact,electrodeColors(cluster_idx,:),contact_size);
    end
end
hold off
    % 'elevation',90,'azimuth',0 -> top
    % 'elevation',0,'azimuth',180 -> front
    % 'elevation',0,'azimuth',-90 -> left
view(0,90)
end