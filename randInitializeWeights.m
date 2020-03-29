function weights = randInitializeWeights(LIn, LOut, epsilonInit)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a nNet layer 
%
%   WEIGHTS = RANDINITIALIZEWEIGHTS(LIn, LOut, epsilonInit) initializes the weights 
%   of a neural network layer with LIn incoming weights and LOut outgoing 
%   weights. Each element is set to a random value in the interval [-epsilonInit,epsilonInit]. 
%   If no specific value for epsilonInit is provided, we set it to sqrt(6)/sqrt(LIn + LOut).
%
%   Input:
%       LIn: Number of ingoing connections to layer.
%       LOut: Number of outgoing connections to layer.
%       epsilonInit (optional): Width of rand distribution for init values. 
%
%   Output:
%       weights: Matrix of random weights of the neural network layer.
%

% Initialize weights 
weights = zeros(LOut, 1 + LIn);

% Set value for epsilon if not given as input parameter
if nargin < 3
    epsilonInit = sqrt(6)/sqrt(LIn + LOut);
end

% Initialize weights by random values
weights = rand(LOut, 1 + LIn) * 2 * epsilonInit - epsilonInit;

end
