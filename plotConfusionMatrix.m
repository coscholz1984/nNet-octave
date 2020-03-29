function hFigure = plotConfusionMatrix (vPredict, vTarget, NUM_LABELS)
%PLOTCONFUSIONMATRIX Plot confusion matrix of a prediction corresponding to a given 
% target from NUM_LABELS classes.
%
%   hFigure = PLOTCONFUSIONMATRIX(vPredict, vTarget, NUM_LABELS) plots a confusion matrix,
%   that is the number of  corresponding predictions and targets in a NUM_LABELS X NUM_LABELS 
%   matrix.
%
%   Input:
%       vPredict: Vector containing the predicted classes for each target.
%       vTarget: Vector containing all target classes, i.e. the ground truth.
%       NUM_LABELS: Number of classes.
%
%   Output:
%       hFigure: Handle to drawn figure.
%
% Created: 2020-03-25

[counts, centers] = hist3([vPredict, vTarget],[NUM_LABELS NUM_LABELS]);
hFigure = figure;
imagesc(counts);
axis image;
xlabel('Target');
ylabel('Prediction');
title('Confusion Matrix');
colorbar;

ticklabels = strsplit(num2str(1:NUM_LABELS));
for iTicks = 1:numel(ticklabels)
  ticklabelsC{iTicks} = ticklabels(iTicks);
end

textStrings = num2str(counts(:), '%i');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:NUM_LABELS);                 % Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center','Color',[.5 .5 .5]);
set(gca, 'XTick', 1:NUM_LABELS, ...              % Change the axes tick marks
         'XTickLabel', ticklabelsC, ...          % and tick labels
         'YTick', 1:NUM_LABELS, ...
         'YTickLabel', ticklabelsC, ...
         'TickLength', [0 0]);

end
