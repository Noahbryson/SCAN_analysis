
%%TODO: make this into a function you monster
function seeg2bipolar(path)
brain = load(sprintf("%s/brain_MNI.mat",path));
electrodeNames = brain.electrodeNames;
electrodeLocs = brain.tala.electrodes;
[rows,cols] = size(electrodeLocs);
bi_names = cell(rows-1,1);
% bipolarLocs = double.empty(locSize);
bi_locs =zeros(rows-1,cols);
for i=1:numel(electrodeNames)-1
    e = electrodeNames{i};
    e1 = electrodeNames{i+1};
    lead = regexp(e,'\D');
    if e(lead) == e1(lead)
        bi_names{i} = sprintf('%s_%s',e,e1(regexp(e1,'\d')));
        bi_locs(i,1) = mean([electrodeLocs(i,1),electrodeLocs(i+1,1)]);
        bi_locs(i,2) = mean([electrodeLocs(i,2),electrodeLocs(i+1,2)]);
        bi_locs(i,3) = mean([electrodeLocs(i,3),electrodeLocs(i+1,3)]);
    end
end
[bi_names,bi_locs] = removeZeroRows(bi_names,bi_locs);








bipolarElectrodes=struct('names',bi_names,'e_loc',bi_locs);
save(sprintf("%s/bipolar_electrodes.mat",path),'bipolarElectrodes')
function [names,locs] = removeZeroRows(names, locs)
    rows2Remove = all(locs==0,2);
    locs(rows2Remove,:) = [];
    names(rows2Remove) = [];
end
end

