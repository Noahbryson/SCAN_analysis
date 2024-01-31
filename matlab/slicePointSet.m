%% slicePointSet.m


x = table2struct(Klesion)

f = fieldnames(x);
for i=1:length(x)
    field = f{i};
    temp = x.(field)
    a.(i)  = temp(10:12)
end


% 
% function slicePointSet(filepath)
% 
% end