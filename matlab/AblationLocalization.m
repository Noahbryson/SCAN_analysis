function fig = AblationLocalization(subject,varargin)
p = inputParser;
opacity = 7; % percent opacity, set to 
addParameter(p,'opacity',opacity)
parse(p,varargin{:})
opacity = p.Results.opacity;
faceAlpha = opacity/100;
path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s",subject);
brainDir = sprintf("%s/brain",path);
load(sprintf("%s/ablation.mat",brainDir));
brain = load(sprintf("%s/brain_MNI.mat",brainDir));
for j=1:length(ablation)
    for i=1:length(brain.electrodeNames)
        loc = ablation(j).sites;
        ele = brain.electrodeNames{i};
        loc_num = regexp(loc,'\d+','match');
        ele_num = regexp(ele,'\d+','match');
        if ele(1) == loc(1) && str2double(loc_num{1}) == str2double(ele_num{1})
            ablation_loc(j) = find(strcmp(brain.electrodeNames, ele));
        end
    end
end

fig = figure;
fig.WindowState = 'maximized';
lab = sprintf("%s %Ablation Location",subject);
fig.Name = lab;
c = [0,0,0];
r = 2;
subplot(1,3,1)
surf = plot3DModel(gca,brain.cortex,brain.annotation.Annotation,"FaceColor",[255,255,255]/255,"FaceAlpha",faceAlpha);
hold on
for i=1:length(ablation_loc)
    contact = brain.tala.electrodes(ablation_loc(i),:);
    % c = [1,1,1];
    % r = 2;
    plotBallsOnVolume(gca,contact,c,r);
end
hold off
view(0,90)
subplot(1,3,2)
surf = plot3DModel(gca,brain.cortex,brain.annotation.Annotation,"FaceColor",[255,255,255]/255,"FaceAlpha",faceAlpha);
hold on
for i=1:length(ablation_loc)
    contact = brain.tala.electrodes(ablation_loc(i),:);
    % c = [0,1,0];
    % r = 2;
    plotBallsOnVolume(gca,contact,c,r);
end
hold off
view(180,0)
subplot(1,3,3)
surf = plot3DModel(gca,brain.cortex,brain.annotation.Annotation,"FaceColor",[255,255,255]/255,"FaceAlpha",faceAlpha);
hold on
for i=1:length(ablation_loc)
    contact = brain.tala.electrodes(ablation_loc(i),:);
    % c = [0,1,0];
    % r = 2;
    plotBallsOnVolume(gca,contact,c,r);
end
hold off
view(-90,0)
end