function [subjectLog, subjectPass] = import_MAYO_data(subject,paths,exportFlag,stimuliFlag)
if nargin < 2 || isempty(paths)
    paths = get_default_paths();
end

if nargin < 3 || isempty(exportFlag)
    exportFlag = false;
end

if nargin < 4 || isempty(stimuliFlag)
    stimuliFlag = true;
end

subject = normalize_subject_name(subject);
subject_ID = strsplit(subject,'-');
subject_ID = subject_ID{1};

subjectLog = struct();
subjectLog.ID = subject;

try
    disp(subject)

    emg_table = readtable(paths.emg_table);
    files = dir(fullfile(paths.input_root,subject,'*mot.mat'));
    if isempty(files)
        error('%s: no mot file found',subject)
    end

    data = load(fullfile(files(1).folder,files(1).name));
    srate = data.srate;

    stim_codes = load_stimuli_from_dat(fullfile(paths.stimuli_root,subject,'samplerun.dat'));
    stim_values = {stim_codes.Value{6,:}}';
    stim_names = cellfun(@(x) strsplit_return_end(x,'_'),stim_values,'UniformOutput',false);
    stim_values = number_stimcodes(stim_names);
    

    ch_desc_fields = fieldnames(data.ch_desc);
    for x=1:length(ch_desc_fields)
        key = ch_desc_fields{x};
        subjectLog.(key) = data.ch_desc.(key);
    end

    brain_file = dir(fullfile(paths.brain_root,subject,'*brain.mat'));
    if isempty(brain_file)
        brain_file = dir(fullfile(paths.brain_root,subject_ID,'*brain.mat'));
    end

    if isempty(brain_file)
        error('%s: no brain file found',subject)
    end

    brain_info = load(fullfile(brain_file(1).folder,brain_file(1).name));
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

    for j=1:size(data.emg,2)
        temp = data.emg(:,j) / 2;
        temp_inv = -1*temp;

        tag=sprintf('%s_1_%d',data.ch_desc.emg_labels{j},idx_counter);
        tag=sanitize_tag(tag,delim);
        signals.(tag) = temp;
        chan_types.(tag) = 'EMG';
        idx_counter=idx_counter+1;

        tag=sprintf('%s_2_%d',data.ch_desc.emg_labels{j},idx_counter);
        tag=sanitize_tag(tag,delim);
        signals.(tag) = temp_inv;
        chan_types.(tag) = 'EMG';
        idx_counter=idx_counter+1;
    end

    states = struct();
    states.StimulusCode = data.stim;

    outdir = fullfile(paths.dump_root,subject_ID);
    session_dir = fullfile(outdir,subject,'preprocessed');

    muscle_mapping = emg_table(contains(emg_table.sub,subject),{'mapping','tag'});
    [muscle_mapping,stim_values] = correct_muscle_mapping(muscle_mapping,stim_values);
    stim_codes.Value(6,:) = stim_values;

    if size(muscle_mapping,1)==0
        error('%s no muscle mapping... likely no EMG detected from input data',subject)
    end

    if exportFlag
        if ~isfolder(session_dir)
            mkdir(session_dir)
        end

        writetable(muscle_mapping, fullfile(outdir,'muscle_mapping.csv'), 'WriteVariableNames', false);
        save(fullfile(session_dir,sprintf('%s.mat',subject)),'signals')
        save(fullfile(session_dir,'states.mat'),'states')
        save(fullfile(session_dir,'channeltypes.mat'),'chan_types')
        parms = struct();
        parms.Stimuli = stim_codes;
        writeStimuliCodes(parms,fullfile(session_dir))
        fID = fopen(fullfile(session_dir,'srate.txt'),'w');
        fprintf(fID,'%d',srate);
        fclose(fID);

        fprintf('exported\n')
    elseif stimuliFlag
        parms = struct();
        parms.Stimuli = stim_codes;
        writeStimuliCodes(parms,fullfile(session_dir))
        fprintf('exported stimuli only\n')
    else
        fprintf('Not Exported\n')
    end

    subjectLog.result='pass';
    fprintf('%s processed successfully\n---------------\n',subject)
catch ME
    report = getReport(ME,"extended","hyperlinks","off");
    warning('USER:ProcessingWarning','%s',report);
    subjectLog.result='fail';
end

subjectPass = build_subject_pass(subjectLog);

end

function paths = get_default_paths()
user = expanduser('~');
paths = struct();
paths.input_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/task_files");
paths.brain_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/imaging");
paths.dump_root = fullfile(user,"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo");
paths.emg_table = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/emg_description.csv");
paths.stimuli_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/raw_data/stimcodes");
end

function subject = normalize_subject_name(subject)
if isstruct(subject)
    subject = subject.name;
elseif isstring(subject)
    subject = char(subject);
end
end

function subjectPass = build_subject_pass(subjectLog)
subjectPass = struct();
subjectPass.ID = subjectLog.ID;
subjectPass.result = subjectLog.result;
subjectPass.logical = strcmp(subjectLog.result,'pass');
end

function tag = sanitize_tag(tag,delim)
tag=strrep(tag,' ',delim);
tag=strrep(tag,'/',delim);
tag=strrep(tag,'-',delim);
end

function stimuli = load_stimuli_from_dat(fp)
fp = char(fp);
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

function [muscle_mapping,stim_values] = correct_muscle_mapping(muscle_mapping,stim_values)
mapping_values = cellstr(muscle_mapping.mapping);
stim_values = cellstr(stim_values);
stim_values = cellfun(@(x) strrep(x,'z',''),stim_values,'UniformOutput',false);
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
