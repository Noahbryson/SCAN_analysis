function writeStimuliCodes(parms,fpath)
stimuli = parms.Stimuli.Value;
stim_codes= cell2table(stimuli(1:end,:));
fpath= strcat(fpath,'/stimuli.mat');
stim_codes = table2struct(stim_codes);
save(fpath,'stim_codes','-mat','-v7');
end