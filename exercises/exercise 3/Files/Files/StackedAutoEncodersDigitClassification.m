%% Training a Deep Neural Network for Digit Classification
% based on the Matlab tutorial
clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();
clc
echo on

% Load the training data into memory
load('digittrain_dataset.mat');

% Display some of the training images
% for i = 1:20
%     subplot(4,5,i);
%     imshow(xTrainImages{i});
% end
echo off
clf
for i = 1:20
    subplot(4,5,i);
    imshow(xTrainImages{i});
end
echo on
%%
% Training the first autoencoder
rng('default')
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)

% Visualizing the weights of the first autoencoder
figure;
plotWeights(autoenc1);

feat1 = encode(autoenc1,xTrainImages);

% Training the second autoencoder
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2,feat1);

% Training the final softmax layer
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
view(softnet)
%% Forming a stacked neural network
view(autoenc1)
view(autoenc2)
view(softnet)

deepnet = stack(autoenc1,autoenc2,softnet);

view(deepnet)

% Get the number of pixels in each image
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Load the test images
load('digittest_dataset.mat');

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
% for i = 1:numel(xTestImages)
%     xTest(:,i) = xTestImages{i}(:);
% end
echo off
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
echo on

%visualise
y = deepnet(xTest);
figure;
plotconfusion(tTest,y);
%% Fine tuning the deep neural network
xTrain = zeros(inputSize,numel(xTrainImages));
% for i = 1:numel(xTrainImages)
%     xTrain(:,i) = xTrainImages{i}(:);
% end
echo off;
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
echo on;

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(xTest);
figure;
plotconfusion(tTest,y);
