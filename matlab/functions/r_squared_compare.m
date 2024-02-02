function r_squared_compare(subject)
path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s",subject);
brainDir = sprintf("%s/brain",path);
dataDir = sprintf("%s/comparisons",path);
brain = load(sprintf("%s/brain_MNI.mat",brainDir));
bipolarElectrodes = load(sprintf("%s/bipolar_electrodes.mat",brainDir)).bipolarElectrodes;

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













































end