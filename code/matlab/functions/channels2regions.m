function [regions,map] = channels2regions(path,labels)
% path: filepath to map.csv file
% lines: lines in the .csv to read. Default is [2,inf]
map = channel_rosa_map(path);
regions = struct;
for i=1:size(map,1)
    regions(i).loc = labels{i};
    regions(i).channel = map{i,3};
    regions(i).rosa = map{i,2};
    regions(i).idx = map{i,1};

end
end