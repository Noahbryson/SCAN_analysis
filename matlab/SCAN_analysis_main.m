close all
subject = "BJH041";
session = "pre_ablation";
% session = "comparisons";
% session = "post_ablation";
% r_squared_compare(subject);
abl = AblationLocalization(subject,'opacity',30);
% r_squared_plots(subject,session,'titleFlag',1,'opacity',30) %top view
% r_squared_plots(subject,session,'elevation',0,'azimuth',-90) %left side view
% r_squared_plots(subject,session,'elevation',0,'azimuth',180) % front facing

% 'elevation',90,'azimuth',0 -> top
% 'elevation',0,'azimuth',180 -> front
% 'elevation',0,'azimuth',-90 -> left
 



