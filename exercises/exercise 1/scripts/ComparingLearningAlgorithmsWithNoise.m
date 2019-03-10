clear
clc
close all

rng(1);
%generation of examples and targets and randomize them using a proportion
x=0:0.05:3*pi; y=sin(x.^2)+normrnd(0,.5);
s = RandStream('mt19937ar','Seed',1);
rand = randperm(s,length(x));
x_rand = x(rand); y_rand = y(rand);
p = con2seq(x_rand); t = con2seq(y_rand); %converting to useful format

proportion = 0.75;
n_training_examples = round(proportion*length(x));

%make test and trian set
train_p = p(1:n_training_examples);
test_p = p(n_training_examples+1:length(x));

train_t = t(1:n_training_examples);
test_t = t(n_training_examples+1:length(x));

%listing different parameters
training_algs = { 'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm', 'trainbr'};
hidden_neurons = [1,5,20,50,200];
i = 0;
j = 0;
time = zeros(length(training_algs),length(hidden_neurons));
rmse_train = time;
rmse_test = time;

for training = training_algs
    i = i+1;
    for n = hidden_neurons
        j=j+1
        
        net = feedforwardnet(n,char(training));
        net.trainParam.epochs=50;
        tic;
        net=train(net,train_p,train_t);
        time(i,j)=toc;
        
        rmse_train(i,j)=sqrt(sum((cell2mat(train_t)-cell2mat(sim(net,train_p))).^2))/length(train_p);
        rmse_test(i,j)=sqrt(sum((cell2mat(test_t)-cell2mat(sim(net,test_p))).^2))/length(test_p);    
    end
    j=0;
end
time
rmse_train
rmse_test

%% Bar Plot Computation time
% returning a barplot for the Speed
names = categorical(training_algs); 
barplot = bar(names, time);
title("Influence of the number of hidden neurons on computation time with noise");
ylabel("Calculation time (s)")
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");
%ylim([0 5]);

%% Bar Plot Training RMSE
% returning a barplot for the RMSE Training Error
names = categorical(training_algs);
barplot = bar(names, rmse_train);
title("Influence of the number of hidden neurons on training error with noise");
ylabel("RMSE per input");
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");

%% Bar Plot Test RMSE
% returning a barplot for the RMSE Test Ernorror names = categorical(training_algs);
barplot = bar(names, rmse_test); title("Influence of the number of hidden neurons on test error with noise"); ylabel("RMSE per input");
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");