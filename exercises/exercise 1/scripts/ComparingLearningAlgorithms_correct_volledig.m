%% Comparing learning algorithms
%generation of examples and targets and randomize them
x=0:0.05:3*pi; y=sin(x.^2);
s = RandStream('mt19937ar','Seed',1);
rand = randperm(s,length(x));
x_rand = x(rand); y_rand = y(rand);

%converting to useful format
p = con2seq(x_rand); t = con2seq(y_rand); 

%dividing data into training and test set
proportion = 0.75;
n_training_examples = round(proportion*length(x));

train_p = p(1:n_training_examples);
test_p = p(n_training_examples+1:length(x));
train_t = t(1:n_training_examples);
test_t = t(n_training_examples+1:length(x));

%listing different parameters
training_algs = { 'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm', 'trainbr'};
hidden_neurons = [1,5,20,50,200];
niterations=10;

%% Without noise

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
        for k = i:niterations
            rng(k);
            net = feedforwardnet(n,char(training));
            net.trainParam.epochs=50;
            tic;
            net=train(net,train_p,train_t);
            time(i,j)=time(i,j)+toc;

            rmse_train(i,j)=rmse_train(i,j)+sqrt(mean((cell2mat(train_t)-cell2mat(sim(net,train_p))).^2));
            rmse_test(i,j)=rmse_test(i,j)+sqrt(mean((cell2mat(test_t)-cell2mat(sim(net,test_p))).^2)); 
        end
        rmse_train(i,j)=rmse_train(i,j)/niterations;
        rmse_test(i,j)=rmse_test(i,j)/niterations;
        time(i,j)=time(i,j)/niterations;
    end
    j=0;
end

% Bar Plot Computation time
figure
names = categorical(training_algs); 
barplot = bar(names, time);
title("Influence of the number of hidden neurons on computation time");
ylabel("Calculation time (s)")
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");

% Bar Plot Training RMSE
figure
names = categorical(training_algs);
barplot = bar(names, rmse_train);
title("Influence of the number of hidden neurons on training error");
ylabel("RMSE");
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");

% Bar Plot Test RMSE
figure
barplot = bar(names, rmse_test); title("Influence of the number of hidden neurons on test error");
ylabel("RMSE");
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");

%% With noise

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
        for k = 1:niterations
            rng(k);
            net = feedforwardnet(n,char(training));
            net.trainParam.epochs=50;
            tic;
            net=train(net,train_p,train_t);
            time(i,j)=time(i,j)+toc;
        
            rmse_train(i,j)=rmse_train(i,j)+sqrt(mean((cell2mat(train_t)-cell2mat(sim(net,train_p))).^2));
            rmse_test(i,j)=rmse_test(i,j)+sqrt(mean((cell2mat(test_t)-cell2mat(sim(net,test_p))).^2)); 
        end
        rmse_train(i,j)=rmse_train(i,j)/niterations;
        rmse_test(i,j)=rmse_test(i,j)/niterations;
        time(i,j)=time(i,j)/niterations;  
    end
    j=0;
end

% Bar Plot Computation time
figure
names = categorical(training_algs); 
barplot = bar(names, time);
title("Influence of the number of hidden neurons on computation time with noise");
ylabel("Calculation time (s)")
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");

% Bar Plot Training RMSE
figure
names = categorical(training_algs);
barplot = bar(names, rmse_train);
title("Influence of the number of hidden neurons on training error with noise");
ylabel("RMSE");
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");

% Bar Plot Test RMSE
figure
barplot = bar(names, rmse_test); title("Influence of the number of hidden neurons on test error with noise"); ylabel("RMSE");
legend_bar = legend(barplot,{"1", "5", "20", "50", "200"});
title(legend_bar, "Amount of hidden neurons");