%% Localize_contacts.m
% output what contacts are in what brain regions, and their associated name in BCI2000/nihon kohden

subjectDir = '/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH046';
mapPath = strcat(subjectDir, '/channel_rosa_map');
rosa_map = channel_rosa_map(mapPath);
VERApath = strcat(subjectDir, '/brain/brain.mat');
data = load(VERApath);
temp = cell(length(rosa_map.rosa),1);
for i=1:length(rosa_map.rosa)
    if data.electrodeNames{i} ~= rosa_map.rosa{i}
        fprintf('misalignment at idx %d\n rosa: %s, map%s',i,data.electrodeNames{i},rosa_map.rosa{i})
        break
    else
        rosa_map.region{i,1} = data.electrodeDefinition.Label{i};
    end
end
disp('aligned')