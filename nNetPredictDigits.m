%% === Neural Network Learning with multi-layer network ========================
%
%  ------------------------------------
%  =========== Introduction ===========
%  ------------------------------------
%  
%  In this example octave code we use a multilayered neural network for pattern 
%  recognition. The task is to recognize handwritten digits by training a 
%  logistic neural network with a sample dataset with known labels.
%  The neural network is optimized on a training dataset and validated on a 
%  validation dataset to find the optimal value of regularization parameter and 
%  prevent overfitting of the training dataset. The performance of the final 
%  dataset is then evaluated by  plotting the confusion matrix and ROC curve 
%  of a test dataset.
%
%  ------------------------------------
%  =========== Dependencies ===========
%  ------------------------------------
%
%  splitDataset.m
%  randInitializeWeights.m
%  nNetCostFunction.m
%  nNetpredict.m
%  fmincg.m
%  plotROC.m
%  plotConfusionMatrix.m
%
% Created: 2020-03-25

% Uncomment next line if you want to start with a clear workspace each time
% clear ; close all; clc

%% =========== Initialization ===========

% === Setup the input parameters and layer sizes === 
INPUT_LAYER_SIZE    = 28 * 28;   % 28x28 Input Images of Digits from EMNIST
HIDDEN_LAYER_SIZE   = [25 19];   % 2 hidden layers with 25 & 19 units
NUM_LABELS          = 10;        % 10 labels, from 0 to 9
options = optimset('MaxIter', 400); % Up to 400 iterations per training epoch

% === Combine input, hidden and output layer size into one vector === 
vLayerSizes = [INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, NUM_LABELS];                         
                          
%% =========== Load and Split Data =============
%  We start the exercise by first loading and splitting the dataset of handwritten digits.
% 
% === Load Data === 
disp('Loading Data ...');
% If not already done, download the dataset first from
% https://www.nist.gov/itl/products-and-services/emnist-dataset
% or via
% urlwrite("http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip","matlab.zip")
% and unzip the file emnist-digits.mat
load('emnist-digits.mat');
% Pick out 5000 images
nNetInput = double(dataset.train.images(1:5000,:))/255;
nNetTarget = dataset.train.labels(1:5000) + 1;

% === Visualize some samples from the dataset ===
pkg load image
hFigure = imageData(nNetInput,25,[28 28]);

% === Split Data into training, validation and test set === 
[dataTrain, dataVal, dataTest, targetTrain, targetVal, targetTest] = splitDataset(nNetInput,nNetTarget,[.5 .3 .2],true);

% === Set up some variables and figures === 
m        = size(dataTrain, 1);
pTrain   = [];
pVal     = [];
hf       = figure;
ax       = gca;
ThetaOut = [];
lOut     = [];

%% =========== Initialize neural network weights ===========

disp('Initializing Neural Network Parameters ...')

for iLayers = 1:numel(vLayerSizes)-1
  initial_Theta{iLayers} = randInitializeWeights(vLayerSizes(iLayers), vLayerSizes(iLayers+1));
end
initialNnetWeights = [];
mThetaSizes = [];
for iLayers = 1:numel(initial_Theta)
  tmpTheta           = initial_Theta{iLayers};
  mThetaSizes        = [mThetaSizes; size(tmpTheta)];
  initialNnetWeights = [initialNnetWeights; tmpTheta(:)];
end

%% ================ Train neural network ================
%
% Here we will train the neural network for differences values of the 
% regularization parameter to find the optimal value.

% === List of regularization parameter values === 
vlambda = [0 0.1 1 5];

% === Loop through values of regularization parameter === 
for lambda = vlambda;  
  % === Create handle to cost function that takes only one input parameter to be minimized === 
  costFunction_ = @(p) nNetCostFunction(p, ...
                                       mThetaSizes, ...
                                       dataTrain, targetTrain, lambda);
  
  fprintf('\nTraining Neural Network...') 

  % === find minimum of nNetWeights for given neural network and training dataset === 
  [nNetWeights, cost] = fmincg(costFunction_, initialNnetWeights, options);
  
  % === Unroll nNetWeights into Thetas again === 
  vThetaSizes = mThetaSizes(:,1)' .* mThetaSizes(:,2)';
  vThetaIndexS   = cumsum([1 vThetaSizes(1:end-1)]);
  vThetaIndexE   = cumsum([vThetaSizes(1:end)]);
  for iLayers = 1:numel(vThetaSizes)
    Theta{iLayers} = reshape(nNetWeights(vThetaIndexS(iLayers):vThetaIndexE(iLayers)),...
                             mThetaSizes(iLayers,1),mThetaSizes(iLayers,2));
  end
  
  % === Calculate target predictions for training and validation data === 
  [predictTrain, hTrain] = nNetPredict(Theta, dataTrain);
  [predictVal, hVal]     = nNetPredict(Theta, dataVal);
  
  % === Calculate accuracy and append === 
  pTrain = [pTrain, mean(double(predictTrain == targetTrain))];
  pVal   = [pVal, mean(double(predictVal == targetVal))];

  % === Print values for current loop === 
  fprintf('\nTraining Set Accuracy: %f\n', pTrain(end) * 100);
  fprintf('Validation Set Accuracy: %f\n', pVal(end) * 100);
  
  % === Update plot of accuracy vs regularization parameter === 
  plot(ax,vlambda(1:numel(pTrain)),pTrain,'o-',vlambda(1:numel(pVal)),pVal,'o-');
  xlabel(ax,'Regularization parameter');
  ylabel(ax,'Accuracy');
  
  % === Store the best performing neural network for later ===
  if numel(pVal) > 1
    if pVal(end) == max(pVal)
      ThetaOut = Theta;
      lOut = lambda;
    end
  else
    ThetaOut = Theta;
    lOut = lambda;
  end
end

% === Calculate predictions on test dataset for best performance nNet === 
[predictTest, hTest] = nNetPredict(ThetaOut, dataTest);

%%  =========== Plot confusion matrix ===========
pkg load statistics
hFigure = plotConfusionMatrix(predictTest, targetTest, NUM_LABELS);

%%  =========== Plot ROC curves ===========
hFigure = plotROC(hTest, targetTest, NUM_LABELS);
