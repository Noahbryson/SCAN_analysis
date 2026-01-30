function signals = writeMATwithHeader(fpath,fname,data,header,removeEmpty)
fpath = strcat(fpath,'/',fname,'.mat');
signals = arrays2Structure(data,header);
if removeEmpty
    vars = fieldnames(signals);
    rm_fields = contains(vars,'EMPTY');
    rm_fields = vars(rm_fields);
    signals = rmfield(signals,rm_fields);
    
end
save(fpath,'signals','-mat','-v7.3');
end