brain = load("/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo/BJH041/brain/brain_MNI.mat");
figure(1)
ax = subplot(2,1,1);
plot3DModel(gca,brain.cortex);
plotBallsOnVolume(gca,brain.tala.electrodes,[[1,234,36]/255],1);
ax = subplot(2,1,2);
plotBallsOnVolume(gca,brain.tala.electrodes,[[1,234,36]/255],1);
hold on
plot3DModel(gca,brain.cortex,brain.annotation.Annotation,"FaceColor",[255,0,0]/255,"FaceAlpha",.2);






