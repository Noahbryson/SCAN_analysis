function labels = makeValidFieldNames(labels)
for i=1:length(labels)
    labels{i} = matlab.lang.makeValidName(labels{i});
end
end