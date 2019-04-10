%% CNN
% Load the digit sample data as an |ImageDatastore| object.

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

%% 
% Display some of the images in the datastore. 
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end


%%
% Check the number of images in each category. 
CountLabel = digitData.countEachLabel;
img = readimage(digitData,1);
size(img)

%%
% Each digit image is 28-by-28-by-1 pixels.
trainingNumFiles = 750;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
				trainingNumFiles,'randomize'); 

%% Define the Network Layers and training phase
% Define the convolutional neural network architecture. 
numFilters=[1,2,5,10,20,35,50,75,100];
filterSize=[2,3,4,5,7,9];%filtersize=[2,3,4,5,7,9,11];
accuracy = zeros(size(numFilters,2),size(filterSize,2));
time=accuracy;
for y = 1:size(numFilters,2)
    numFilters(y);
    for z = 1:size(filterSize,2)
       filterSize(z);
      
%        layers = [imageInputLayer([28 28 1])
%            convolution2dLayer(filterSize(z),numFilters(y))
%            reluLayer
%            maxPooling2dLayer(2,'Stride',2)
%            fullyConnectedLayer(10)
%            softmaxLayer
%            classificationLayer()];  %+-3min

        layers = [imageInputLayer([28 28 1])
          convolution2dLayer(filterSize(z),numFilters(y))
          reluLayer

          maxPooling2dLayer(2,'Stride',2)

          convolution2dLayer(filterSize(z),numFilters(y)*2)
          reluLayer  

          fullyConnectedLayer(10)
          softmaxLayer
          classificationLayer()];

        options = trainingOptions('sgdm','MaxEpochs',15, ...
            'InitialLearnRate',0.0001);  
        tic
        convnet = trainNetwork(trainDigitData,layers,options);
        time(y,z)=toc

        YTest = classify(convnet,testDigitData);
        TTest = testDigitData.Labels;

        % Calculate the accuracy. 
        accuracy(y,z) = sum(YTest == TTest)/numel(TTest)   
    end
end
accuracy
time