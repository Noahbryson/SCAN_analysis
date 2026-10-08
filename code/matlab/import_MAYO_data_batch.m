user = expanduser('~');

paths = struct();
paths.input_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/task_files");
paths.brain_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/imaging");
paths.dump_root = fullfile(user,"Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo");
paths.emg_table = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/emg_description.csv");
paths.stimuli_root = fullfile(user,"Documents/NCAN/projects/inter-effectors/SCAN_MAYO_DATA/raw_data/stimcodes");

exportFlag = false;
stimuliFlag = true;
subjects = dir(fullfile(paths.input_root,'Mayo*'));
%%
subjectLogCell = cell(1,length(subjects));
subjectPass = repmat(struct('ID','','result','','logical',false),1,length(subjects));

for i=1:length(subjects)
    subject = subjects(i).name;
    [subjectLogCell{i},subjectPass(i)] = import_MAYO_data(subject,paths,exportFlag,stimuliFlag);
end

subjectLog = align_struct_fields(subjectLogCell);

function struct_array = align_struct_fields(struct_cells)
if isempty(struct_cells)
    struct_array = struct();
    return
end

all_fields = {};
for i=1:length(struct_cells)
    current_fields = fieldnames(struct_cells{i});
    for j=1:length(current_fields)
        if ~ismember(current_fields{j},all_fields)
            all_fields{end+1} = current_fields{j};
        end
    end
end

empty_values = cell(1,length(all_fields));
empty_values(:) = {[]};
struct_array = repmat(cell2struct(empty_values,all_fields,2),1,length(struct_cells));

for i=1:length(struct_cells)
    current_fields = fieldnames(struct_cells{i});
    for j=1:length(current_fields)
        struct_array(i).(current_fields{j}) = struct_cells{i}.(current_fields{j});
    end
end
end
logout = fullfile(paths.dump_root,'Mayo_raw_conversion_log.csv');
table_log = struct2table(subjectPass);
writetable(table_log,logout);