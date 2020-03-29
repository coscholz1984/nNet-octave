function [xTrain, xVal, xTest, yTrain, yVal, yTest] = splitDataset(mInput, vTarget, vRatios, bRnd)
%SPLITDATASET splits a dataset into three parts based on given ratios.
%
%   [xTrain, xVal, xTest, yTrain, yVal, yTest] = splitDataset(mInput,vTarget,vRatios,bRnd) splits
%   a dataset given by predictors mInput and targets vTarget into training, validation and test 
%   sets. The relative sizes of each set are given by a 3x1 vector vRatios. If bRnd is 
%   true then the set will be randomly shuffeled before splitting the data.
%
%   Input:
%       mInput: Matrix of data to image with dimension M x N, where N is the linear image size.
%       vTarget: Vector containing the targets for each entry in the input data.
%       vRatios: 3x1 vector containing ratios in which the dataset should be split into.
%       bRnd(optional): Setting this to true will shuffle the data before splitting.
%
%   Output:
%       xTrain, xVal, xTest: Training, validation and test dataset.
%       yTrain, yVal, yTest: Training, validation and test targets.
%

% Initialize return variables and arguments
xTrain = [];
xVal = [];
xTest = [];
if nargin < 4
  bRnd = false;
end
if numel(vRatios) ~= 3
  disp('Length of vRatios must be 3.');
  return
end
if sum(vRatios) ~= 1
  disp('vRatios not normalized, normalizing now.');
  vRatios = vRatios/sum(vRatios);
end

nI = size(mInput,1);
nxTrain = round(vRatios(1)*nI);
nxVal = round(vRatios(2)*nI);
nxTest = round(vRatios(3)*nI);

% Distribute targets randomly in training, validation and test data
if bRnd
  vRndIndex = randperm(nI); 
  mInput = mInput(vRndIndex,:);
  vTarget = vTarget(vRndIndex); 
end

% Make sure that data is split such that nxTrain + nxVal + nxTest == nI
while nI - nxTrain - nxVal - nxTest ~= 0
  npick = randi(3,1);
  switch npick
    case 1
      nxTrain = nxTrain - sign(nI - nxTrain - nxVal - nxTest);
    case 2
      nxVal = nxVal - sign(nI - nxTrain - nxVal - nxTest);
    case 3
      nxTest = nxTest - sign(nI - nxTrain - nxVal - nxTest);
  end
end

xTrain = mInput(1:nxTrain,:);
xVal = mInput(nxTrain+(1:nxVal),:);
xTest = mInput(nxTrain+nxVal+(1:nxTest),:);

yTrain = vTarget(1:nxTrain);
yVal = vTarget(nxTrain+(1:nxVal));
yTest = vTarget(nxTrain+nxVal+(1:nxTest));

end
