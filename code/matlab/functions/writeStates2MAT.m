function states = writeStates2MAT(fpath,states)
fpath = strcat(fpath,'/states','.mat');
save(fpath,'states','-mat','-v7');
end