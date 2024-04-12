function output = channel_rosa_map(filename, dataLines)
%IMPORTFILE Import data from a text file
%  CHANNELROSAMAP = IMPORTFILE(FILENAME) reads data from text file
%  FILENAME for the default selection.  Returns the data as a table.
%
%  CHANNELROSAMAP = IMPORTFILE(FILE, DATALINES) reads data for the
%  specified row interval(s) of text file FILENAME. Specify DATALINES as
%  a positive scalar integer or a N-by-2 array of positive scalar
%  integers for dis-contiguous row intervals.
%
%  Example:
%  channelrosamap = importfile("C:\Users\nbrys\Box\Brunner Lab\DATA\SCAN_Mayo\BJH046\channel_rosa_map.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 02-Feb-2024 15:34:32

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 3, "Encoding", "UTF-8");

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["IDX", "Rosa", "LABEL"];
opts.VariableTypes = ["double", "char", "char"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Rosa", "LABEL"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Rosa", "LABEL"], "EmptyFieldRule", "auto");

% Import the data
channelrosamap = readtable(filename, opts);
channelrosamap = table2struct(channelrosamap);
rosa = cell(length(channelrosamap),1);
idx = zeros(length(channelrosamap),1);
lab = cell(length(channelrosamap),1);
for i=1:length(channelrosamap)
    t = channelrosamap(i).Rosa;
    comps = strsplit(t(2:end-1),"'");
    if length(comps) > 1
        rosa{i} = strcat(comps{1},"'",comps{2});
    else
        rosa{i} = strcat(comps{1});
        % rosa{i} = channelrosamap(i).Rosa;
    end
    lab{i} = channelrosamap(i).LABEL;
    idx(i) = channelrosamap(i).IDX;
end
output = struct;
output.idx = idx;
output.rosa = rosa;
output.label = lab;
end