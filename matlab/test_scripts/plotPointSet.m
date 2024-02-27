

subject = 'BJH045';
session = 'aggregate';

path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s",subject);
brainDir = sprintf("%s/brain",path);
dataDir = sprintf("%s/%s/analyzed",path,session);
brain = load(sprintf("%s/brain.mat",brainDir));
fp = sprintf("%s/bipolar_electrodes.mat",brainDir);
if ~isfile(fp)
seeg2bipolar(brainDir)
end
bipolarElectrodes = load(sprintf("%s/bipolar_electrodes.mat",brainDir),'bipolarElectrodes').bipolarElectrodes;
[regions, map] = channels2regions(sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s/channel_rosa_map.csv",subject),brain.SecondaryLabel);
simple_chans = table2cell(map(:,end));
bi_names = cell(size(simple_chans));
for i=1:numel(simple_chans)-1
    e = char(simple_chans{i});
    e1 = char(simple_chans{i+1});
    lead = regexp(e,'\D');
    if e(lead) == e1(lead)
        bi_names{i} = sprintf('%s_%s',e,e1(regexp(e1,'\d')));
    end
end
 bi_names = bi_names(~cellfun('isempty',bi_names));
 bi_names = cell2struct(bi_names,'ref',2);
