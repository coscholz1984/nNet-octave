function [p, hout] = nNetPredict(Theta, nNetInput)
%NNETPREDICT Predict the label of an input given a trained neural network with 
%weights Theta.
%
%   [p, hout] = NNETPREDICT(Theta, nNetInput) outputs the predicted label of nNetInput given the
%   trained weights of a neural network, where Theta is a cell array containing 
%   the layer weights.
%
%   Input:
%       Theta: Cell array of weights of neural network in 2nd order tensor 
%              notation.
%       nNetInput: Data on which the prediction will be calculated.
%
%   Output:
%       p: Predicted class.
%       hout: Matrix of probabilities for each class.
%
% Created: 2020-03-25

nInput = size(nNetInput, 1);
num_labels = size(Theta{end}, 1);

% Initialize p 
p = zeros(size(nNetInput, 1), 1);

% Calculate probabilities and find maximum
h{1} = logSigmoid([ones(nInput, 1) nNetInput] * Theta{1}');
for iTheta = 2:numel(Theta)
  h{iTheta} = logSigmoid([ones(nInput, 1) h{iTheta-1}] * Theta{iTheta}');
end
hout = h{end};
[dummy, p] = max(hout, [], 2);

end

function vSig = logSigmoid(vIn)
%SIGMOID Compute logistic sigmoid function
%   VSIG = LOGSIGMOID(z) computes the logistic sigmoid of vIn. 
%   This function will accept vIn as a matrix or a vector.
%

vSig = 1.0 ./ (1.0 + exp(-vIn));
end
