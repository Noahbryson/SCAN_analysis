%% get ROIs
subject = 'BJH046';
path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s",subject);
brainDir = sprintf("%s/brain",path);
ROIs = {'insula','postcentral','precentral',};
brain = load(sprintf("%s/brain.mat",brainDir));
[regions, map] = channels2regions(sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s/channel_rosa_map.csv",subject),brain.SecondaryLabel);
