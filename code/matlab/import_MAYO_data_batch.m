
user = expanduser('~'); % Get local path for interoperability on different machines, function in my tools dir.
input_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/task_files");
brain_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/imaging");
dump_root= fullfile(user,"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo"); % Path to data
emg_table = readtable('/Users/nkb/Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/emg_description.csv');
tab_out = "/Users/nkb/Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/stim_codes.mat";
load(tab_out) % yields variable stim_codes, written to all folders for stimuli indexing.

% %%
% template_dat = struct();
% load('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH071/day1_run1_L/preprocessed/BJH071.mat');
% load('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH071/day1_run1_L/preprocessed/channeltypes.mat');
% load('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH071/day1_run1_L/preprocessed/states.mat');
% load('/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH071/day1_run1_L/preprocessed/stimuli.mat');
% template_dat.signals = signals;
% template_dat.chan_types=chan_types;
% template_dat.states=states;
% template_dat.stim_codes=stim_codes;
%%


exportFlag=false;

% EMG_remap = struct(); EMG_remap.hand = 'wristExtensor'; EMG_remap.foot = 'TBA'; EMG_remap.tongue = 'tongue';
task_pattern = '_mot.mat';
subjects = dir(fullfile(input_root,'Mayo*'));
emg_tags = struct();
emg_loc = 1;
subjectLog = struct();
for i=1:length(subjects)
    try
    files = dir(fullfile(input_root,subjects(i).name,'*mot.mat'));
    subject = subjects(i).name;
    subject_ID = strsplit(subject,'-');
    subject_ID = subject_ID{1};
    subjectLog(i).ID = subject;
    disp(subject)
    data = load(fullfile(files.folder,files.name));
    srate = data.srate;

    stimuli_root = fullfile(user,'Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/raw_data/stimcodes');
    stim_codes = load_stimuli_from_dat(fullfile(stimuli_root,subject,'samplerun.dat'));
    
    stim_values = {stim_codes.Value{6,:}}';
    stim_names = cellfun(@(x) strsplit_return_end(x,'_'),stim_values,'UniformOutput',false);
    stim_values = number_stimcodes(stim_names);
    stim_codes.Value(6,:) = stim_values;
    
    


    
    for x=1:length(fieldnames(data.ch_desc))
        key = fieldnames(data.ch_desc);
        key = key{x};
        subjectLog(i).(key) = data.ch_desc.(key);
    end
    brain_file = dir(fullfile(brain_root,subjects(i).name,'*brain.mat'));
    if length(brain_file)<1
        
        brain_file = dir(fullfile(brain_root,subject_ID,'*brain.mat'));
    end
    brain_info = load(fullfile(brain_file.folder,brain_file.name));

    data_array = data.data;
    
        labels = brain_info.lbls; 
        labels = labels(data.ch_desc.brain);
        signals = struct();
        chan_types = struct();
        idx_counter = 1;
        delim = 'xxx';

        for j=1:size(data_array,2)
            tag=sprintf('%s_%d',labels{j},idx_counter);
            tag=strrep(tag,' ',delim);
            tag=strrep(tag,'-',delim);
            signals.(tag) = data_array(:,j);
            chan_types.(tag) = 'sEEG';
            idx_counter=idx_counter+1;
        end

        for j=1:size(data.emg,2) % split EMG to be re-referenced in my pipeline, just a standardization convention nothing wild.
            temp = data.emg(:,j);
            temp = temp / 2;
            temp_inv = -1*temp;
            % emg half 1
            
            tag=sprintf('%s_1_%d',data.ch_desc.emg_labels{j},idx_counter);
            tag=strrep(tag,' ',delim);
            tag=strrep(tag,'/',delim);
            tag=strrep(tag,'-',delim);



            emg_tags(emg_loc).sub = subject;
            emg_tags(emg_loc).tag = tag;
            emg_tags(emg_loc).ch_name = data.ch_desc.emg_labels{j};
            emg_loc = emg_loc + 1;
            try
                signals.(tag) = temp;
                chan_types.(tag) = 'EMG';
                idx_counter=idx_counter+1;

                % emg half 2
                tag=sprintf('%s_2_%d',data.ch_desc.emg_labels{j},idx_counter);
                tag=strrep(tag,' ',delim);
                tag=strrep(tag,'/',delim);
                tag=strrep(tag,'-',delim);
                signals.(tag) = temp_inv;
                chan_types.(tag) = 'EMG';
                idx_counter=idx_counter+1;
            catch ME
                disp(ME.message)
                error('%s: EMG Error',subject);
            end

        end
        states = struct();
        states.StimulusCode = data.stim;

        % export signals, stim_codes, states, chan_types, muscle_mapping
        
        outdir = fullfile(dump_root,subject_ID);
        session_dir = fullfile(outdir,subject,'preprocessed');
        
        muscle_mapping = emg_table(contains(emg_table.sub,subject),{'mapping','tag'});
        muscle_mapping = correct_muscle_mapping(muscle_mapping,stim_values);
        
        % muscle_mapping = emg_table(contains(emg_table.sub,subject),{'mapping','ch_name'});
        if size(muscle_mapping,1)==0
            error('%s no muscle mapping... likely no EMG detected from input data',subject)
        else
            
            if exportFlag
                if ~isfolder(session_dir)
                    mkdir(session_dir)
                end
                writetable(muscle_mapping, fullfile(outdir,'muscle_mapping.csv'), 'WriteVariableNames', false); %muscle mapping output
                save(fullfile(session_dir,sprintf('%s.mat',subject)),'signals')
                save(fullfile(session_dir,'states.mat'),'states')
                save(fullfile(session_dir,'channeltypes.mat'),'chan_types')
                save(fullfile(session_dir,'stimuli.mat'),'stim_codes')
                fID = fopen(fullfile(session_dir,'srate.txt'),'w');
                fprintf(fID,'%d',srate);
                fclose(fID);
                
                fprintf('exported\n')
            else
                fprintf('Not Exported\n')
            end
            subjectLog(i).result='pass';
            fprintf('%s processed successfully\n---------------\n',subject)
        end
        

    catch ME
        formatStr = sprintf('%s failed, not exported',subject);
        report = getReport(ME,"extended","hyperlinks","off");
        warning('USER:ProcessingWarning','%s',report);
        subjectLog(i).result='fail';
    end






end

subjectPass= struct('ID',{subjectLog(:).ID},'result',{subjectLog(:).result});
res_log = isequal({subjectLog(:).result},'pass');
subjectPass(:).logical = res_log;
%%
function stimuli = load_stimuli_from_dat(fp)
[~,~,params] = load_bcidat(fp);
stimuli = params.Stimuli;
end

function outStr = strsplit_return_end(inStr,delim)


x = strsplit(inStr,delim);
outStr=x{end};

end

function values = number_stimcodes(stim_codes)
values = stim_codes;
for i=1:length(stim_codes)
    values{i} = sprintf('%d_%s',i,stim_codes{i});
end

end

function muscle_mapping = correct_muscle_mapping(muscle_mapping,stim_values)
mapping_values = cellstr(muscle_mapping.mapping);
stim_values = cellstr(stim_values);
stim_names = cellfun(@(x) strsplit_return_end(x,'_'),stim_values,'UniformOutput',false);

for i=1:length(mapping_values)
    if ismember(mapping_values{i},stim_values)
        continue
    end

    map_name = strsplit_return_end(mapping_values{i},'_');
    stim_match = strcmp(map_name,stim_names);

    if sum(stim_match)==1
        mapping_values{i} = stim_values{stim_match};
    elseif sum(stim_match)==0
        error('Could not match muscle mapping value "%s" to any stimulus value',mapping_values{i});
    else
        error('Muscle mapping value "%s" matched multiple stimulus values',mapping_values{i});
    end
end

muscle_mapping.mapping = mapping_values;

end
