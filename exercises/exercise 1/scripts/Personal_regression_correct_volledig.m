%% preparing data
%loading the data
load('Data_Problem1_regression.mat');

%student number r0595714
%5 highest numbers; 9 7 5 5 4
Tnew = (9*T1+7*T2+5*T3+5*T4+4*T5)/(9+7+5+5+4);

% random single generator
s = RandStream('mt19937ar','Seed',1); 

% random indices
rand = randperm(s,length(Tnew));

% shuffle input and output
Tnew_shuffled = Tnew(rand);
X1_shuffled = X1(rand);
X2_shuffled = X2(rand);

% construct training/validation and test set
Ttrain = Tnew_shuffled(1:1000);
Xtrain = [X1_shuffled(1:1000) X2_shuffled(1:1000)];

Tvalidate = Tnew_shuffled(1001:2000);
Xvalidate = [X1_shuffled(1001:2000) X2_shuffled(1001:2000)];

Ttest = Tnew_shuffled(2001:3000);
Xtest = [X1_shuffled(2001:3000) X2_shuffled(2001:3000)] ;

%% 3D scatterplot of training set
c = linspace(2,9,length(X1));
c=c(rand);
c=c(1:1000);

scatter3(X1_shuffled(1:1000),X2_shuffled(1:1000),Ttrain,[],c,'.')
xlabel("input X1");
ylabel("input X2");
zlabel("target training set");
title("3d plot of target with regard to inputs of the training set");

%% Experiment
% deciding which training algorithm and number of hidden neurons to take
training_algs = { 'trainlm', 'trainbr','trainbfg'};
hidden_neurons = [25,50,100,200];
i = 0;
j = 0;
time = zeros(length(training_algs),length(hidden_neurons));
rmse_train = time;
rmse_test = time;

niteration=10;

for training = training_algs
    i = i+1;
    for n = hidden_neurons
        j=j+1;
        for k = 1:niteration
            
            rng(k);
            net = feedforwardnet(n,char(training));

            % In order to fully use the training set, as training set
            net.divideParam.trainRatio = 1; 
            net.divideParam.valRatio = 0; 
            net.divideParam.testRatio = 0;
            
            net.trainParam.epochs=50;
            
            tic;
            net.trainParam.showWindow = false;
            net=train(net,Xtrain.',Ttrain.');
            time(i,j)=time(i,j)+toc;
        
            rmse_train(i,j)=rmse_train(i,j)+sqrt(mean((Ttrain.'-sim(net,Xtrain.')).^2));
            rmse_test(i,j)=rmse_test(i,j)+sqrt(mean((Tvalidate.'-sim(net,Xvalidate.')).^2));    
        end
        time(i,j)=time(i,j)/niteration;
        rmse_train(i,j)=rmse_train(i,j)/niteration;
        rmse_test(i,j)=rmse_test(i,j)/niteration;
    end
    j=0;
end

% Bar Plot Computation time
figure
names = categorical(training_algs); 
barplot = bar(names, time);
title("Influence of the number of hidden neurons on computation time");
ylabel("Calculation time (s)")
legend_bar = legend(barplot,{"25", "50", "100", "200"});
title(legend_bar, "Amount of hidden neurons");

% Bar Plot Training RMSE
figure
names = categorical(training_algs);
barplot = bar(names, rmse_train*1000);
title("Influence of the number of hidden neurons on training error");
ylabel("RMSE (x10^-^3)");
legend_bar = legend(barplot,{"25", "50", "100", "200"});
title(legend_bar, "Amount of hidden neurons");

% Bar Plot Test RMSE
figure
barplot = bar(names, rmse_test*1000); title("Influence of the number of hidden neurons on validation error");
ylabel("RMSE (x10^-^3)");
legend_bar = legend(barplot,{"25", "50", "100", "200"});
title(legend_bar, "Amount of hidden neurons");

%% Experiment 
% 2 hidden layers

training_algs = {'trainbr'};
hidden_neurons = [5,10,15,20,25];
i = 0;
j = 0;
time = zeros(length(training_algs),length(hidden_neurons));
rmse_train = time;
rmse_test = time;

for training = training_algs
    i = i+1;
    for n = hidden_neurons
        j=j+1;
        fprintf('%d \n',j);
        for k = i:niteration
        
            net = feedforwardnet([n n],char(training));           
            % In order to fully use the training set, as training set
            net.divideParam.trainRatio = 1; 
            net.divideParam.valRatio = 0; 
            net.divideParam.testRatio = 0;
            
            net.trainParam.epochs=50;
     
            tic;
            net.trainParam.showWindow = false;
            net=train(net,Xtrain.',Ttrain.');
            time(i,j)=time(i,j)+toc;
     
            rmse_train(i,j)=rmse_train(i,j)+sqrt(mean((Ttrain.'-sim(net,Xtrain.')).^2));
            rmse_test(i,j)=rmse_test(i,j)+sqrt(mean((Tvalidate.'-sim(net,Xvalidate.')).^2));    
        end
        time(i,j)=time(i,j)/niteration;
        rmse_train(i,j)=rmse_train(i,j)/niteration;
        rmse_test(i,j)=rmse_test(i,j)/niteration;
    end
    j=0;
end

% Bar Plot
figure
tests=["computation time (s)";"RMSE training set (x10^-^2)";"RMSE validation set (x10^-^2)"];
names = categorical(tests); 
barplot = bar(names,[time; rmse_train*100; rmse_test*100]);
title("Influence of the two hidden layers for bayesian regularization");
legend_bar = legend(barplot,{"5","10","15","20","25"});
title(legend_bar, "Hidden neurons \newline per layer");

fprintf('done');

%% Experiment
% 3 hidden layers

training_algs = {'trainbr'};
hidden_neurons = [3,5,10,15,20];
i = 0;
j = 0;
time = zeros(length(training_algs),length(hidden_neurons));
rmse_train = time;
rmse_test = time;

for training = training_algs
    i = i+1;
    for n = hidden_neurons
        j=j+1;
        fprintf('%d \n',j);
        for k = i:niteration
        
           net = feedforwardnet([n n n],char(training));           
            % In order to fully use the training set, as training set
            net.divideParam.trainRatio = 1; 
            net.divideParam.valRatio = 0; 
            net.divideParam.testRatio = 0; 
            
            net.trainParam.epochs=50;
     
            tic;
            net.trainParam.showWindow = false;
            net=train(net,Xtrain.',Ttrain.');
            time(i,j)=time(i,j)+toc;
     
            rmse_train(i,j)=rmse_train(i,j)+sqrt(mean((Ttrain.'-sim(net,Xtrain.')).^2));
            rmse_test(i,j)=rmse_test(i,j)+sqrt(mean((Tvalidate.'-sim(net,Xvalidate.')).^2));    
        end
        time(i,j)=time(i,j)/niteration;
        rmse_train(i,j)=rmse_train(i,j)/niteration;
        rmse_test(i,j)=rmse_test(i,j)/niteration;
    end
    j=0;
end

% barplot
figure
tests=["computation time (s)";"RMSE training set (x10^-^2)";"RMSE validation set (x10^-^2)"];
names = categorical(tests); 
barplot = bar(names,[time; rmse_train*100; rmse_test*100]);
title("Influence of the three hidden layers for bayesian regularization");
legend_bar = legend(barplot,{"3","5","10","15","20"});
title(legend_bar, "Hidden neurons per layer");

fprintf('done');
%% Experiment
% Different activation functions
functions = {'logsig','radbas','radbasn','tansig'};
hidden_neurons = [100];

time = zeros(length(functions),length(hidden_neurons));
rmse_train = time;
rmse_test = time;
i = 0;
j = 0;

for func = functions
    i = i+1;
    for n = hidden_neurons
        j=j+1;
        fprintf("%d \n",j);
        for k = 1:niteration
            
            rng(k);
            net = feedforwardnet(n,'trainbr');
            
            net.layers{1}.transferFcn = char(func);
            net.trainParam.epochs=50;
            
            % In order to fully use the training set, as training set
            net.divideParam.trainRatio = 1; 
            net.divideParam.valRatio = 0; 
            net.divideParam.testRatio = 0;
                  
            tic;
            net.trainParam.showWindow = false;
            net=train(net,Xtrain.',Ttrain.');
            time(i,j)=time(i,j)+toc;
        
            rmse_train(i,j)=rmse_train(i,j)+sqrt(mean((Ttrain.'-sim(net,Xtrain.')).^2));
            rmse_test(i,j)=rmse_test(i,j)+sqrt(mean((Tvalidate.'-sim(net,Xvalidate.')).^2));    
        end
        time(i,j)=time(i,j)/niteration;
        rmse_train(i,j)=rmse_train(i,j)/niteration;
        rmse_test(i,j)=rmse_test(i,j)/niteration;
    end
    j=0;
end

% plot
figure
tests=["computation time";"RMSE training set *(x10^-^3)";"RMSE validation set *(x10^-^3)"];
names = categorical(tests); 
barplot = bar(names,[time.'; rmse_train.'*1000; rmse_test.'*1000]);
title("Effect of different activation functions");
legend_bar = legend(barplot,{ 'logsig','radbas','radbasn','tansig'});
title(legend_bar, "Activation functions");

fprintf('done');

%% performance assessment
% 3d scatterplot of test set's targets and network output

    rng(1);
    net = feedforwardnet(100,'trainbr');

    % In order to fully use the training set, as training set
    net.divideParam.trainRatio = 1; 
    net.divideParam.valRatio = 0; 
    net.divideParam.testRatio = 0;

    net.trainParam.epochs=50;

    net=train(net,Xtrain.',Ttrain.');

    mean((Ttrain.'-sim(net,Xtrain.')).^2)
    mean((Tvalidate.'-sim(net,Xvalidate.')).^2)
    mean((Ttest.'-sim(net,Xtest.')).^2)

    a=sim(net,Xtest.');
    figure
    c = linspace(5,8,length(X1));
    c=c(rand);
    c=c(2001:3000);

    scatter3(X1_shuffled(2001:3000),X2_shuffled(2001:3000),Ttest,35,c,'.')
    hold on
    scatter3(X1_shuffled(2001:3000),X2_shuffled(2001:3000),a,10,'black')
    hold off
    xlabel("input X1");
    ylabel("input X2");
    zlabel("output/target");
    title("3d plot target and output for the test set")
    
% error curve
    % using the previous approach, we can't generate an error curve from
    % nntraintool, therefor we have to change the approach and data
    rng(1)
    net = feedforwardnet(100, 'trainbr');
    
    net.trainParam.epochs = 50; 
    net.divideParam.trainRatio = 1/3; 
    net.divideParam.valRatio = 1/3; 
    net.divideParam.testRatio = 1/3;
    
    input = [X1_shuffled(1:3000) X2_shuffled(1:3000)];
    target = Tnew_shuffled(1:3000);
    [net, tr] = train(net,input.',target.');