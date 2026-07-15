function channels = writeChannelDescriptions(fpath,channels,chan_types,removeEmpty)
fpath = strcat(fpath,'/channeltypes','.mat');
chan_types = arrays2Structure(chan_types,channels);
if removeEmpty
    vars = fieldnames(chan_types);
    rm_fields = contains(vars,'EMPTY');
    rm_fields = vars(rm_fields);
    chan_types = rmfield(chan_types,rm_fields);
end
save(fpath,'chan_types','-mat','-v7');
end