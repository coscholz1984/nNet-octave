# nNet-octave
Octave implementation of logistic multilayer neural network

Example of a neural network with multiple logistic neuron 
layers. The neural network is trained and used to predict numbers 
from handwritten digits from the EMNIST dataset.
---------------------------------------------------------------------

The code is based on the content taught in the Stanford Machine 
learning course by Andrew Ng, found at 
https://www.coursera.org/learn/machine-learning

For numerical minimization of the cost function the routine 
fmincg.m by Carl Edward Rasmussen is used, where use is permitted 
for purposes of research or education (see copyright notice within 
fmincg.m).

The dataset used in this example is the handwritten digits set 
from EMNIST:
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
    EMNIST: an extension of MNIST to handwritten letters.

The dataset can be downloaded in mat-format from
https://www.nist.gov/itl/products-and-services/emnist-dataset

In this example we will use a multi-layer neural network, train 
for several regularization parameter values and pick the best 
performing candiate. Then this optimized network is characterized 
on a test dataset using confusion matrix and ROC curve.

The code has been tested and run in an GNU Octave 4.4.1 
environment. The code uses the image and statistics packages for 
displaying data. The actual neural network does not require any 
non-core packages.

This code is intendend for educational purpose only. All steps 
required for neural network classification are implemented from 
scratch in such a way that it is hopefully easy enough to 
understand for those who want to get familiar with the architec-
ture of neural networks. When redistributing this code, which is 
only granted for educational porpuse you are required to cite all 
sources from this document, retain this notice and make note of 
any changes that have been made.

This implementation is not equally performant as many other 
neural network implementations available for several platforms 
and should not be used for performance-critical projects. 

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Auhtor: Christian Scholz, March 2020

Summary of attched scripts
---------------------------------------------------------------------

nNetPredictDigits.m
-----------------------
This is the main script where we demonstrate a multilayered 
neural network for pattern recognition. The task is to recognize 
handwritten digits by training a logistic neural network with a 
sample dataset with known labels.

nNetPredict.m
-----------------------
Predict the label of an input given a trained neural network 
with given weights.

nNetCostFunction.m
-----------------------
Calculates the neural network cost function for a multilayer 
logistic neural network for classification.

randInitializeWeights.m
-----------------------
Randomly initialize the weights of a nNet layer. Each 
element is set to a random value in the interval 
[-epsilonInit,epsilonInit].

splitDataset.m
-----------------------
Splits a dataset into three parts (training, validation and 
test set) based on given ratios.

imageData.m
-----------------------
Plot a set of nImages from a dataset X, where each image has 
size vSize.

plotConfusionMatrix.m
-----------------------
Plot confusion matrix of a prediction corresponding to a given 
target from NUM_LABELS classes.

plotROC.m
-----------------------
Plot ROC curve of a prediction probability to a given  target 
from NUM_LABELS classes.

fmincg.m
-----------------------
Minimize a continuous differentialble multivariate function. 
Authored by Carl Edward Rasmussen.
