%% Image Category Classification Using Deep Learning
% Download Image Data
% Download the compressed data set from the following location
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes', 'ferry', 'laptop'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)
%%
% Because |imds| above contains an unequal number of images per category,
% let's first adjust it, so that the number of images in the training set
% is balanced.

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

% Find the first instance of an image for each category
airplanes = find(imds.Labels == 'airplanes', 1);
ferry = find(imds.Labels == 'ferry', 1);
laptop = find(imds.Labels == 'laptop', 1);

figure
subplot(1,3,1);
imshow(imds.Files{airplanes})
subplot(1,3,2);
imshow(imds.Files{ferry})
subplot(1,3,3);
imshow(imds.Files{laptop})

%% Download Pre-trained Convolutional Neural Network (CNN)
% Location of pre-trained "AlexNet"
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
% Store CNN model in a temporary folder
cnnMatFile = fullfile(tempdir, 'imagenet-caffe-alex.mat');

if ~exist(cnnMatFile, 'file') % download only once     
    disp('Downloading pre-trained CNN model...');     
    websave(cnnMatFile, cnnURL);
end

% Load Pre-trained CNN
% Load MatConvNet network into a SeriesNetwork
convnet = helperImportMatConvNet(cnnMatFile)

% View the CNN architecture
convnet.Layers

% Inspect the first layer
convnet.Layers(1)

% Inspect the last layer
convnet.Layers(end)

% Number of class names for ImageNet classification task
numel(convnet.Layers(end).ClassNames)

%% Pre-process Images For CNN
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Prepare Training and Test Image Sets

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

% Get the network weights for the second convolutional layer
w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')
