%student number r0595714
%5 highest numbers; 9 7 5 5 4

Tnew = (9*T1+7*T2+5*T3+5*T4+4*T5)/(9+7+5+5+4);

%Using three different random generators
[s1,s2,s3] = RandStream.create('mrg32k3a','NumStreams',3);

rand = randperm(s1,length(Tnew));
Tnew_rand = Tnew(rand);
trainl = Tnew_rand(1:1000)
X1_rand = X1(rand);
X1train = X1_rand(1:1000)
X2_rand = X2(rand);
X2train = X2_rand(1:1000)

%for the color in the plot
c = linspace(2,9,length(X1));
c=c(rand);
c=c(1:1000);

scatter3(X1train,X2train,trainl,[],c,'.')
xlabel("input X1");
ylabel("input X2");
zlabel("target training set");
title("3d plot of target with regard to inputs of the training set")


rand = randperm(s2,length(Tnew));
Tnew_rand = Tnew(rand);
validatel = Tnew_rand(1:1000)
X1_rand = X1(rand);
X1validate = X1_rand(1:1000)
X2_rand = X2(rand);
X2validate = X2_rand(1:1000)

rand = randperm(s3,length(Tnew));
Tnew_rand = Tnew(rand);
testl = Tnew_rand(1:1000)
X1_rand = X1(rand);
X1test = X1_rand(1:1000)
X2_rand = X2(rand);
X2test = X2_rand(1:1000)


training_algs = { 'trainlm', 'trainbr'};
hidden_neurons = [25,50,100,200,400];
i = 0;
j = 0;
time = zeros(length(training_algs),length(hidden_neurons));
rmse_train = time;
rmse_test = time;
rng(1);

for training = training_algs
    i = i+1;
    for n = hidden_neurons
        j=j+1
        
        net = feedforwardnet(n,char(training));
        net.trainParam.epochs=50;
        tic;
        net=train(net,[X1train X2train].',trainl.');
        time(i,j)=toc;
        
        rmse_train(i,j)=sqrt(sum(trainl.'-sim(net,[X1train X2train].')).^2);
        rmse_test(i,j)=sqrt(sum(validatel.'-sim(net,[X1validate X2validate].')).^2);    
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
title("Influence of the number of hidden neurons on computation time");
ylabel("Calculation time (s)")
legend_bar = legend(barplot,{"25", "50", "100", "200", "400"});
title(legend_bar, "Amount of hidden neurons");

%% Bar Plot Training RMSE
% returning a barplot for the RMSE Training Error
names = categorical(training_algs);
barplot = bar(names, rmse_train);
title("Influence of the number of hidden neurons on training error");
ylabel("RMSE");
legend_bar = legend(barplot,{"25", "50", "100", "200", "400"});
title(legend_bar, "Amount of hidden neurons");

%% Bar Plot Test RMSE
% returning a barplot for the RMSE Test Ernorror names = categorical(training_algs);
barplot = bar(names, rmse_test); title("Influence of the number of hidden neurons on validation error"); ylabel("RMSE");
legend_bar = legend(barplot,{"25", "50", "100", "200", "400"});
title(legend_bar, "Amount of hidden neurons");

