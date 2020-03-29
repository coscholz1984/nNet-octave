function hFigure = plotROC (hPredict, vTarget, NUM_LABELS)
%PLOTROC Plot ROC curve of a prediction probability to a given 
% vTarget from NUM_LABELS classes.
%
%   hFigure = PLOTROC(hPredict, vTarget, ) plots a ROC curve,
%   that is the number of true positives vs false positives at a given decision 
%   threshold.
%
%   Input:
%       hPredict: Matrix containing probabilities for all predictions.
%       vTarget: Vector containing all target classes per datapoint.
%       NUM_LABELS: Number of possible classes.
%
%   Output:
%       hFigure: Handle to figure drawn.
%
% Created: 2020-03-25

rocThresh = fliplr([0:0.001:1]);

hFigure = figure;
hold on;
for iClass = 1:NUM_LABELS;
  plot(sum((hPredict(:,iClass) > rocThresh) & ~(vTarget == iClass))/sum(~(vTarget == iClass)), ...
       sum((hPredict(:,iClass) > rocThresh) & (vTarget == iClass))/sum(vTarget == iClass), ...
       '.-','DisplayName',['Target class ' num2str(iClass)]);
end
legend('Location','southeast');
hold off;
xlabel('False Positives Rate');
ylabel('True Positives Rate');
title('ROC curve');

end
