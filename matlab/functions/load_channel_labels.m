function output = load_channel_labels(filepath,rosa_map)
%LOAD_CHANNEL_LABELS Summary of this function goes here
%   take a .mat file with channel labels and identifiers for colored
%   indexing to show groups. 

labels = load(filepath);
fields = fieldnames(labels);
channelName = cell(size(fields));
channelLabel = zeros(size(fields));
chanSig = zeros(size(fields));
SCANflag = zeros(size(fields));
veraName = cell(size(fields));
veraIdx = zeros(size(fields));
for i=1:length(fields)
name = fields{i};
name = erase(name,"_");
veraLoc = strfind(rosa_map.label,name);
Index = find(not(cellfun('isempty',veraLoc)));
channelName{i}  = name;
channelLabel(i) = labels.(fields{i})(1);
chanSig(i) = labels.(fields{i})(2);
SCANflag(i) = labels.(fields{i})(3);
veraName{i} = rosa_map.rosa{Index(1)};
veraIdx(i) = Index(1);
end
output = struct();
output.channel = channelName;
output.channelClass = channelLabel;
output.significant = chanSig;
output.vera = veraName;
output.vera_idx = veraIdx;
output.SCANflag = SCANflag;
end

