%% PCA
%% Redundancy and random data
data = randn(100,10);
load choles_all
data = p.'
%%
cov(data)
[v,d]=eig(cov(data))
max(diag(d))
for i =1:5
[a,b]=eigs(cov(data),i);
fraction = (sum(diag(b))/sum(diag(d)))
rmse = sum(sqrt(mean(data*a*transpose(a)-data).^2)); % is zero of all information is captured by dimensionality reduction
cov(data)-v*d*v^-1; % is 0, should be zero
end
sort(diag(d));

%%
[y1,PS] = mapstd(data)
processpca(y1.',0.08)
%% handwritten digits
load threes -ascii
colormap('gray')
imagesc(reshape(threes(1,:),16,16),[0,1])
imagesc(reshape(mean(threes),16,16),[0,1])
axis off
size(cov(threes))
[v,d]=eig(cov(threes))
%plot eigenvalues
toplot = sort(diag(d),'descend')
bar(toplot)
xlim([0,100])
xlabel("Number of PCA");
ylabel("Eigenvalue");
title("Bar plot of eigenvalues of mean three")
%% compressing data
for i =1:4
[a,b]=eigs(cov(threes),i);
fraction = (sum(diag(b))/sum(diag(d)))
reconstructed=threes*a*transpose(a);
rmse = sum(sqrt(mean(reconstructed-threes).^2))
%colormap('gray')
subplot(2,2,i)
imagesc(reshape(mean(reconstructed),16,16),[0,1])
title(i + " PCA('s), variance explained: " + round((fraction * 100),2) + "%");
axis off;
end
%% plotting reconstruction errors
rmse_=zeros(50,1);
for i =1:50
[a,b]=eigs(cov(threes),i);
fraction = (sum(diag(b))/sum(diag(d)))
reconstructed=threes*a*transpose(a);
rmse_(i) = mean(sqrt(mean(reconstructed-threes).^2))
end
figure
plot(rmse_)
xlabel("Accumulated number of PCA");
ylabel("MRMSE");
title("Reconstruction error for different numbers of PCA's")
%%
[a,b]=eigs(cov(threes),256);
fraction = (sum(diag(b))/sum(diag(d)))
reconstructed=threes*a*transpose(a);
rmse = mean(sqrt(mean(reconstructed-threes).^2))
%%
rmse_=zeros(256,1);
for i =1:256
[a,b]=eigs(cov(threes),i);
fraction = (sum(diag(b))/sum(diag(d)))
reconstructed=threes*a*transpose(a);
rmse_(i) = mean((mean(reconstructed-threes).^2))
end
a=cumsum(diag(d))
a=sort([0;a(1:(end-1))],'descend')
plot(a)
hold on
plot(rmse_*300)
hold off
xlabel("Number of PCA");
title("Connection between reconstruction error and left out eigenvalues")
legend("Accumulated eigenvalues not in reconstruction","MMSE (x300)");