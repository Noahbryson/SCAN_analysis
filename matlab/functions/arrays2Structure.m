function out = arrays2Structure(data,labels)
labels = makeValidFieldNames(labels);
for j=1:size(data,2)
    out.(labels{j}) = data(:,j);
end
end