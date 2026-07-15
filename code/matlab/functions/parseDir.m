function targets = parseDir(dir,target,exclude)
targets = {};
for i=1:length(dir)
    if (contains(dir(i).name,target)) && (~contains(dir(i).name,exclude))
        targets{end+1} = dir(i).name;
    end
end
end