function [fCost vGrad] = nNetCostFunction(nNetWeights, ...
                                     mThetaSizes, ...
                                     nNetInput, nNetTarget, lambda)
%NNETCOSTFUNCTION calculates the neural network cost function for a multilayer
%logistic neural network for classification
%
%   [fCost vGrad] = NNETCOSTFUNCTON(nNetWeights, mThetaSizes, nNetInput, nNetTarget, lambda) 
%   computes the cost fCost and cost gradient vGrad of the neural network in 
%   terms of the weights.  
% 
%   Input:
%       nNetWeights: Weights of all neural network layers rolled into one vector.
%       mThetaSizes: Matrix that contains the size of each layer when unrolled into 
%                    2nd order tensor notation.
%       nNetInput: Data used for calculating the cost on.
%       nNetTarget: Targets corresponding to data in nNetInput.
%       lambda: Regularization parameter, penalizing the total weight of the network.
%               This can be used to prevent overfitting.
%
%   Output:
%       fCost: Value of the cost function for given input.
%       vGrad: Gradient of the cost function for given input in the directions of all 
%              continuous parameters (i.e. the weights).
%

vThetaSizes = mThetaSizes(:,1)' .* mThetaSizes(:,2)';

vThetaIndexS = cumsum([1 vThetaSizes(1:end-1)]);
vThetaIndexE = cumsum([vThetaSizes(1:end)]);

% Reshape nNetWeights back into the Theta parameters
for iLayers = 1:numel(vThetaSizes)
  Theta{iLayers} = reshape(nNetWeights(vThetaIndexS(iLayers):vThetaIndexE(iLayers)),...
                           mThetaSizes(iLayers,1),mThetaSizes(iLayers,2));
end
       
% Initialize return arguments 
fCost = 0;
for iTheta = 1:numel(Theta)
    ThetaGrad{iTheta} = zeros(size(Theta{iTheta}));
end

% Initialize variables
nInput = size(nNetInput, 1);
% Set a{1} to nNetInput
a{1} = nNetInput;
for iTheta = 1:numel(Theta)
  a{iTheta}   = [ones(size(a{iTheta},1),1) a{iTheta}];
  a{iTheta+1} = logSigmoid(a{iTheta}*Theta{iTheta}');
end

% Add bias to input layer
nNetInput = [ones(nInput,1) nNetInput];

% Reshape nNetTarget so that it uses labels per index instead of numbers
index = [1:length(nNetTarget)'; nNetTarget']';
nNetTargetL = zeros(length(nNetTarget),mThetaSizes(end,1));
nNetTargetL(sub2ind(size(nNetTargetL),index(:,1),index(:,2)))=1;

% Compute cost function
for iTheta = 1:numel(Theta)
  tTheta = Theta{iTheta};
  fCost      = fCost + sum(sum(tTheta(:,2:end).^2));
end
fCost = 1/nInput * sum(sum(-nNetTargetL.*log(a{end})-(1-nNetTargetL).*log(1-a{end}))) + lambda/(2*nInput) * (fCost);

% Perform backpropagration to calculate gradient
delta = cell(numel(Theta),1);
delta{end} = a{end} - nNetTargetL;
for iTheta = 1:numel(Theta)-1
  delta_tmp         = delta{end-iTheta+1} * Theta{end-iTheta+1} .* [ones(size(a{end-iTheta-1}, 1),1) ...
                                                                    logSigmoidGradient(a{end-iTheta-1}*Theta{end-iTheta}')];
  delta{end-iTheta} = delta_tmp(:,2:end);
end

for iTheta=1:numel(Theta)
    ThetaGradTmp = 1/nInput * delta{iTheta}'*a{iTheta};
    ThetaTmp               = Theta{iTheta};
    ThetaGradTmp(:,2:end) = ThetaGradTmp(:,2:end) + lambda/nInput*ThetaTmp(:,2:end);
    ThetaGrad{iTheta}      = ThetaGradTmp;
end

% Combine gradients to output vector
vGrad = [];
for iGrad = 1:numel(ThetaGrad)
  vGrad = [vGrad; ThetaGrad{iGrad}(:)];
end

end

function vSig = logSigmoid(vIn)
%SIGMOID Compute logistic sigmoid function
%
%   VSIG = LOGSIGMOID(z) computes the logistic sigmoid of vIn. 
%   This function will accept vIn as a matrix or a vector.
%

vSig = 1.0 ./ (1.0 + exp(-vIn));
end

function vGrad = logSigmoidGradient(vIn)
%SIGMOIDGRADIENT returns the gradient of the logistic sigmoid function
%evaluated at vIn
%
%   VGRAD = SIGMOIDGRADIENT(VIN) computes the gradient of the sigmoid function
%   evaluated at vIn. This will accept vIn as a matrix or a vector. 
%

vGrad = zeros(size(vIn));
vGrad = logSigmoid(vIn).*(1-logSigmoid(vIn));
end
