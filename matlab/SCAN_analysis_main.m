clf
subject = "BJH041";
session = "pre_ablation";
% session = "comparisons";
% session = "post_ablation";
% r_squared_compare(subject);
% abl = AblationLocalization(subject,'opacity',20,'forVideo',1,'full',0);
path = sprintf("C:/Users/nbrys/Box/Brunner Lab/DATA/SCAN_Mayo/%s/gifs",subject);
% for i=1:3
% cond = i;
conditions = r_squared_plots(subject,session,'forVideo',0,'conditionIdx',cond,'opacity',20,'full',0,'titleFlag',1) %top view
fname = sprintf('%s_%s',session,conditions{cond})
% animation3D(path,fname,0,120,3)
% close all
% end
% r_squared_plots(subject,session,'elevation',0,'azimuth',-90) %left side view
% r_squared_plots(subject,session,'elevation',0,'azimuth',180) % front facing


 



