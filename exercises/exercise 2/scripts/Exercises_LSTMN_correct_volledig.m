%% 2.2 Neural network
testd = load('laserpred.dat');
traind = load('lasertrain.dat');

mu = mean(traind);
sig = std(traind);

traind = (traind - mu) / sig;
testd = (testd - mu) / sig;

hidden_neurons = [1,2,5,10,25,50];
lags = [1,5,10,25,50,100,200,500,999];
time = zeros(length(lags),length(hidden_neurons));
rmse = time;
numberit=50;
for y = 1:size(lags,2)
    y
    for z = 1:size(hidden_neurons,2)
        hidden_neurons(z)
        tic;
        rng(1);
        net = feedforwardnet(hidden_neurons(z),'trainbr');        
        % In order to fully use the training set, as training set
        net.divideParam.trainRatio = 1; 
        net.divideParam.valRatio = 0; 
        net.divideParam.testRatio = 0; 
        net.trainParam.epochs=50;
        net.trainParam.showWindow = false;

        [TrainData,TrainTarget]=getTimeSeriesTrainData(traind, lags(y));
        net = configure(net,TrainData,TrainTarget);
        net = train(net,TrainData,TrainTarget);

        % - RNN implementation
        % initialize data
        inputForPrediction=traind((end-lags(y)+1):end);

        % predict the future
        rmseval = 0;
        for i = 1:numberit
            simres = sim(net, inputForPrediction);
            inputForPrediction = [inputForPrediction(2:end);simres];
            rmseval=rmseval+sqrt(mean((testd(i)-simres).^2));
        end
        rmse(y,z)=rmseval; %RMSE of standardized data
        time(y,z)=toc;
    end
end
rmse=rmse/numberit;
mesh(hidden_neurons,lags,rmse.^-1)
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
xlim([1,50])
ylim([0,1000])
title('1/RMSE')
xlabel('Number of hidden neurons')
ylabel('Lag')
%% for 5 hidden neurons and a lag of 50, minimum in RMSE reached (4.3591)
% for 25 hidden neurons and a lag of 100, also low rmse (5.5437)
tic;
lag = 100;%100
hidden = 25;%25
% visualise results
rng(1);
net = feedforwardnet(hidden,'trainbr');
% In order to fully use the training set, as training set
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net.trainParam.epochs=250;
%net.trainParam.showWindow = false;

[TrainData,TrainTarget]=getTimeSeriesTrainData(traind, lag);
net = configure(net,TrainData,TrainTarget);
net = train(net,TrainData,TrainTarget);

% - RNN implementation
% initialize data
inputForPrediction=traind((end-lag+1):end);
predictions=zeros(100,1);
% predict the future
rmseval = 0;
numberit=100;
for i = 1:numberit
    %inputForPrediction
    simres = sim(net, inputForPrediction);
    predictions(i)=simres;
    inputForPrediction = [inputForPrediction(2:end);simres];
    rmseval=rmseval+sqrt(mean((testd(i)-simres).^2));
end
rmseval = rmseval/numberit;
toc
plot(predictions,'b')
hold on
plot(testd,'g')
title('lag = 50, hidden neurons = 5')%title('lag = 100, hidden neurons = 25')
xlabel('time step')
legend('predicted', 'true')
%% 2.3 LSTMN
%https://nl.mathworks.com/help/deeplearning/examples/time-series-forecasting-using-deep-learning.html
testd = load('laserpred.dat');
traind = load('lasertrain.dat');

mu = mean(traind);
sig = std(traind);

traind = (traind - mu) / sig;
testd = (testd - mu) / sig;

%Prepare Predictors and Responses
xtrain = traind(1:end-1);
ytrain = traind(2:end);

hidden_neurons = [1,2,5,10,25,50,75,100,125,150,175,200,250,300,350,400];
number_epochs = [50,100,150,200,250,300];

rmse_ = zeros(length(number_epochs),length(hidden_neurons));
numFeatures = 1;
numResponses = 1;
for y = 1:size(number_epochs,2)
    n_epoch = number_epochs(y)
    for z = 1:size(hidden_neurons,2)
        numHiddenUnits = hidden_neurons(z)
        rng(1);
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        options = trainingOptions('adam', ...
            'MaxEpochs',n_epoch, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.01, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',100, ...
            'LearnRateDropFactor',0.4, ...
            'Verbose',0);
        net = trainNetwork(xtrain.',ytrain.',layers,options);
        net = predictAndUpdateState(net,xtrain.');
        a = testd(end);
        [net,YPred] = predictAndUpdateState(net,a);

        numTimeStepsTest = numel(testd);
        for i = 2:numTimeStepsTest
            [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
        end
        rmse_(y,z) = sqrt(mean((YPred-testd.').^2)) %RMSE of standardized data
        %rmse_(y,z) = sqrt(mean((YPred(1:50)-testd(1:50).').^2)) %RMSE of standardized data
    end
end
%% Mesh figure
mesh(hidden_neurons,number_epochs,rmse.^-1)
set(gca, 'XScale', 'log')
xlim([1,400])
ylim([0,300])
title('1/RMSE')
xlabel('Number of hidden neurons')
ylabel('Number of epochs')

%% Best result of experiment
numHiddenUnits = 150; %2
n_epoch=250; %300
rng(1);
tic;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',n_epoch, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.4, ...
    'Verbose',0);
net = trainNetwork(xtrain.',ytrain.',layers,options);
net = predictAndUpdateState(net,xtrain.');
a = testd(end);
[net,YPred] = predictAndUpdateState(net,a);

numTimeStepsTest = numel(testd);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
sqrt(mean((YPred(1:50)-testd(1:50).').^2)) %RMSE of standardized data
toc
figure
plot(traind(1:end-1))
hold on
idx = size(traind,1):(size(traind,1)+numTimeStepsTest);
plot(idx,[traind(end) YPred],'.-')
hold off
xlabel("Time step")
ylabel("Standardized value")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(testd)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Standardized value")
title("Forecast")

subplot(2,1,2)
stem(YPred - testd.')
xlabel("Time step")
ylabel("Error")
%title("RMSE = " + sqrt(mean((YPred(1:100)-testd(1:100).').^2)))
title("RMSE = " + sqrt(mean((YPred(1:50)-testd(1:50).').^2)))