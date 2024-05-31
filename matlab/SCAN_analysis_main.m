%%
addpath(genpath('/Users/nkb/Documents/NCAN/code/MATLAB_tools'))
addpath(genpath('/Users/nkb/Documents/NCAN/code/SCAN_analysis/'))
%% Subject Identification
rootPath = '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo';
subject = "BJH041";
session = "pre_ablation";
% session = "comparisons";
% session = "post_ablation";

path = sprintf("/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo//%s",subject);
brainDir = sprintf("%s/brain",path);
dataDir = sprintf("%s/%s/analyzed",path,session);
brain = load(sprintf("%s/brain_MNI.mat",brainDir));
[annotation_remap,cmap,name,name_id] = createColormapFromAnnotations(brain.annotation);
brain_cmap = vertex_cmap(cmap,annotation_remap);
target_cmap = parseColorMap(brain.annotation,{'precentral','postcentral','paracentral'});
target_cmap = vertex_cmap(target_cmap,annotation_remap);

rosa_map = channel_rosa_map('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/channel_rosa_map.csv');
colors=[62,108,179; 27,196,225; 129,199,238;44,184,149;0,129,145;193,189,47;200,200,200]/256;
%% Task Metric Analysis
% r_squared_compare(subject);
% abl = AblationLocalization(subject,'opacity',20,'forVideo',1,'full',0);
path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s/gifs",subject);
% for i=1:3
% cond = i;
conditions = r_squared_plots(subject,session,'forVideo',0,'conditionIdx',1,'opacity',20,'full',0,'titleFlag',1); %top view
fname = sprintf('%s_%s',session,conditions{cond});
% animation3D(path,fname,0,120,3)
% close all
% end
% r_squared_plots(subject,session,'elevation',0,'azimuth',-90) %left side view
% r_squared_plots(subject,session,'elevation',0,'azimuth',180) % front facing

%% Clustering Results
close all
figpath = "/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/Posters/ASSFN2024/figures/Inter-effector Localization";
title = 'axial';
%%
% [brain,target_cmap] = sliceBrain(brain,'l',target_cmap);
cluster_labels = load_channel_labels(fullfile(rootPath,subject ,session,'analyzed/clustering_result/BJH041_pre_ablation_r_sq_(r^2)_5_clustered_channel_labels.mat'),rosa_map);
plot_electrode_labels(brain,cluster_labels,colors,target_cmap,'opacity',5,'full',1);
lab = sprintf('5_%s.eps',title);
% saveas(gcf,fullfile(figpath,lab),'epsc')
%%
cluster_labels = load_channel_labels(fullfile(rootPath,subject ,session,'analyzed/clustering_result/BJH041_pre_ablation_r_sq_(r^2)_3_clustered_channel_labels.mat'),rosa_map);
plot_electrode_labels(brain,cluster_labels,colors(3:end,:),target_cmap,'opacity',5,'full',1,'SCANonly',1);
lab = sprintf('3_%s.eps',title);
% saveas(gcf,fullfile(figpath,lab),'epcs')
%%
close all
figure(4);
[BRAIN,tcmap] = sliceBrain(brain,'l',target_cmap);
precentral = getbrainregion('precentral',BRAIN);

% surf = plot3DModel(gca,brain.cortex,brain.annotation.Annotation,...
%     "FaceVertexCData",target_cmap,"FaceColor",'interp',"FaceAlpha",0.05);
surf = plot3DModel(gca,BRAIN.cortex,BRAIN.annotation.Annotation, ...
    "FaceVertexCData",tcmap,"FaceColor",'interp',"FaceAlpha",0.15);
hold on
plot3DModel(gca,precentral.cortex,precentral.annotation.Annotation,...
    "FaceVertexCData",precentral.annotation.cmap,"FaceAlpha",0.5);

view(-90,15)

hold off
%%
close all
plot3DModel(gca,precentral.cortex,precentral.annotation.Annotation,"FaceVertexCData",precentral.annotation.cmap,"FaceAlpha",0.15);
hold on
target_contacts = {'ML3','ML8','JL13','KL11'};
for i=1:length(target_contacts)
    electrode_loc = strfind(rosa_map.label,target_contacts{i});
    idx = find(not(cellfun('isempty',electrode_loc)));
    contact = BRAIN.tala.electrodes(idx,:);
    plotBallsOnVolume(gca,contact,colors(i+1,:),3);
end
hold off
view(-90,15)

%%
function X = vertex_cmap(cmap,annotation_remap)
X = zeros(size(annotation_remap,1),3);

for i=1:numel(annotation_remap)
    X(i,:) = cmap(annotation_remap(i),:);
end
end

function output_cmap = parseColorMap(annotation,regions)
[annotation_remap,cmap,name,name_id] = createColormapFromAnnotations(annotation);
output_cmap = ones(size(cmap))*0.95;
for i=1:length(regions)
    loc = strfind(name,regions{i});
    idx = find(not(cellfun('isempty',loc)));
    output_cmap(idx,:) = cmap(idx,:);
end


end

function [BRAIN,CMAP] = sliceBrain(brain,side,cmap)
if side == 'l'
    side = 1;
end
if side =='r'
    side = 2;
end
vertLog = brain.cortex.vertId == side;
triLog = brain.cortex.triId == side;
BRAIN = brain;
BRAIN.cortex.tri = brain.cortex.tri(triLog,:);
BRAIN.cortex.vert = brain.cortex.vert(vertLog,:);
CMAP = 0;
BRAIN.annotation.Annotation = brain.annotation.Annotation(vertLog);
if (exist('cmap','var'))
CMAP = cmap(vertLog,:);
end

end

function region = getbrainregion(region_name,wholeBrain)
region = struct();
identifier = strfind({wholeBrain.annotation.AnnotationLabel(:).Name},region_name)';
idx = find(not(cellfun('isempty',identifier)));
identifier = wholeBrain.annotation.AnnotationLabel(idx).Identifier;
locs = wholeBrain.annotation.Annotation == identifier;
targetVerts = find(locs==1);
vertRemap = linspace(1,length(targetVerts),length(targetVerts))';
region.cortex.vert = wholeBrain.cortex.vert(locs,:);
membership = ismember(wholeBrain.cortex.tri(:,:),targetVerts);
rowHasMembers = all(membership,2);
mapobj = containers.Map(targetVerts,vertRemap);
tris = wholeBrain.cortex.tri(rowHasMembers,:);
tris_out = zeros(size(tris));
for i=1:size(tris,1)
    for j=1:size(tris,2)
        tris_out(i,j) = mapobj(tris(i,j));
    end
end
region.cortex.tri = tris_out;
region.annotation.Annotation = wholeBrain.annotation.Annotation(locs);
region.annotation.color = wholeBrain.annotation.AnnotationLabel(idx).PreferredColor;
region.annotation.cmap = ones(size(region.annotation.Annotation,1),3)*region.annotation.color;

    
end