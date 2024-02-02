function animation3D(fpath,fname,elevation,numFrames,frameSpeed)
% Set up the figure properties

set(gca, 'nextplot', 'replacechildren', 'Visible', 'off');

% Number of frames in the animation
% Preallocate the struct array for the frames
frames(numFrames) = struct('cdata', [], 'colormap', []);

% Rotate the view and capture frames
for i = 1:numFrames
    % Rotate the view (azimuth, elevation)
    view(i*frameSpeed, elevation); % Adjust as needed
    axis equal;
    axis manual;
    % Capture the frame
    frames(i) = getframe(gcf);
end

% Save the frames as an animated GIF
filename = sprintf('%s/%s.gif',fpath,fname); % Specify the output file name
% for i = 1:numFrames
%     [imind, cm] = rgb2ind(frame2im(frames(i)), 256);
%     if i == 1
%         imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
%     else
%         imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
%     end
% end
for i = 1:numFrames
    [imind, cm] = rgb2ind(frame2im(frames(i)), 256);

    % Find the index of the background color in the colormap
    % Assuming the background color is white, which is [1 1 1]
    [~, background_color_index] = ismember([0 1 0], cm, 'rows');

    if i == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, ...
                'DelayTime', 0.01, 'TransparentColor', background_color_index);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', ...
                'DelayTime', 0.01, 'TransparentColor', background_color_index);
    end
end











end
