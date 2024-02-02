function writeCSVwithHeader(fpath,fname,data,header,removeEmpty)
fpath = strcat(fpath,'/',fname,'.csv');
dataTable = array2table(data);
dataTable.Properties.VariableNames = header;
if removeEmpty
    vars = dataTable.Properties.VariableNames;
    rm_cols = contains(vars,'EMPTY');
    dataTable(:,rm_cols) = [];
end
writetable(dataTable, fpath, 'WriteVariableNames', true);
end