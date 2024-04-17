function X = detectThresholdCrossing(data, threshold, distance)
%   [...] = FINDPEAKS(...,'MinPeakHeight',MPH) finds only those peaks that
%   are greater than the minimum peak height, MPH. MPH is a real-valued
%   scalar. The default value of MPH is -Inf.
%   [...] = FINDPEAKS(...,'MinPeakDistance',MPD) finds peaks separated by
%   more than the minimum peak distance, MPD. This parameter may be
%   specified to ignore smaller peaks that may occur in close proximity to
%   a large local peak. For example, if a large local peak occurs at LOC,
%   then all smaller peaks in the range [N-MPD, N+MPD] are ignored. If not
%   specified, MPD is assigned a value of zero. 
locs = data > threshold;
data(locs) = 1;
[~,X] = findpeaks(data,'MinPeakDistance',distance,'MinPeakHeight',threshold);
