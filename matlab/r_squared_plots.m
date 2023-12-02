
function r_squared_plots(subject,session)
path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s",subject);
brainDir = sprintf("%s/brain",path);
dataDir = sprintf("%s/%s/analyzed",path,session);
brain = load(sprintf("%s/brain_MNI.mat",brainDir));
bipolarElectrodes = load(sprintf("%s/bipolar_electrodes.mat",brainDir)).bipolarElectrodes;
% List all .mat files in the directory
matFiles = dir(fullfile(dataDir, '*.mat'));
% Initialize an empty structure
r_sq_res = struct();
% Loop through each file and load its contents into the structure
for k = 1:length(matFiles)
    % Extract the base file name (without extension)
    [~, baseFileName, ~] = fileparts(matFiles(k).name);

    % Load the file
    loadedData = load(fullfile(dataDir, matFiles(k).name));
    names = fieldnames(loadedData);
    fieldValues = zeros(length(names), 1);
    for i = 1:length(names)
        fieldValues(i) = loadedData.(names{i});
    end
    for i=1:numel(names)
        out(i).name = names{i};
        out(i).r_sq = fieldValues(i);
    end
    % Assign the loaded data to a field in the structure
    % The field name is the base file name
    r_sq_res.(baseFileName(3:end)) = out;
end
if session == "post_ablation"
    load(sprintf("%s/ablation.mat",brainDir));
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
end



conditions = fieldnames(r_sq_res);
fig = figure;
fig.WindowState = 'maximized';
fig.Name = sprintf("%s %s",subject,session);
for j=1:length(conditions)
    subplot(3,1,j)

    surf = plot3DModel(gca,brain.cortex,brain.annotation.Annotation,"FaceColor",[255,255,255]/255,"FaceAlpha",.2);
    hold on
    locs = bipolarElectrodes.e_loc;
    for i=1:length(r_sq_res.(conditions{j}))
        res = r_sq_res.(conditions{j})(i).r_sq;
        contact = locs(i,:);
        intensity = abs(res);
        colorFlag = res > 0;
        if colorFlag
            rgb = [1,0,0];
        else
            rgb = [0,0,1];
        end
        plotBallsOnVolume(gca,contact,rgb,intensity*3+1);
    end
    if session == "post_ablation"
        for i=1:length(ablation_loc)
            contact = brain.tala.electrodes(ablation_loc(i),:);
            c = [0,1,0];
            r = 1;
            plotBallsOnVolume(gca,contact,c,r);
        end

    end
    title(conditions{j})
    hold off
end