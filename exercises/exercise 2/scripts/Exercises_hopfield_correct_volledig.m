%% exercise 1

T = [1 1 1; -1 -1 1; 1 -1 -1]';

  input1={[1;1;1]}
  input2={[-1;-1;1]}
  input3={[1;-1;-1]}
  %unwishedAttractor={[-1;1]}
  
%T = [+1 -1 +1; ...
%      +1 -1 -1];  
  %input1={[1;1]}
  %input2={[-1;-1]}
  %input3={[1;-1]}
  %unwishedAttractor={[-1;1]}

plot(T(1,:),T(2,:),'r*')
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');

net = newhop(T);
truecounter=0;
firstseen=zeros(1,100);
for i = 1:100
    a = {[0;-80;0]};
    %a = {rands(2,1)};
    [y,Pf,Af] = sim(net,{1 100},{},a);
    for j = 1:100
       if isequal(y(:,j),input1)==1
           truecounter=truecounter+1;
           firstseen(j)=firstseen(j)+1;
           %celldisp(y(:,j))
           break
       else if isequal(y(:,j),input2)==1
               truecounter=truecounter+1;
               firstseen(j)=firstseen(j)+1;
               %celldisp(y(:,j))
               break
           else if isequal(y(:,j),input3)==1
                   truecounter=truecounter+1;
                   firstseen(j)=firstseen(j)+1;
                   %celldisp(y(:,j))
                   break  
               else
                   if j==100
                       %if isequal(y(:,j),unwishedAttractor)==0
                            celldisp(y(:,j))
                       %end
                   end
               end
           end
       end
    end
end
firstseen 
truecounter %7518
bar(firstseen(1:60))
title('Iteration after which a desired attractor is reached, for 10000 tests')
xlabel('Iteration')
ylabel('Number of occurrences')
%% exercise 2

noiselevels = [0,0.1,0.2,0.4,0.75,1,2,4,7.5,10,20,40,75,100,200,400];
num_iter=[1,2,4,8,15,25,50,75,100,150,200,300,400,500];
correctnumbers = zeros(size(noiselevels,2),size(num_iter,2));
spuriousstates = zeros(size(noiselevels,2),size(num_iter,2));
for j=1:10
    j
    for i=1:size(num_iter,2)
        i
        [correctnumber,nSpuriousStates]=hopdigit_v3(noiselevels(1,j),num_iter(1,i));
        correctnumbers(j,i)=correctnumber;
        spuriousstates(j,i)=nSpuriousStates;
    end
end
correctnumbers
spuriousstates
mesh(num_iter,noiselevels,correctnumbers)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
xlim([1,500])
ylim([0,400])
title('Number of correctly recovered digits')
xlabel('number of iterations')
ylabel('noise level')
mesh(num_iter,noiselevels,spuriousstates)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
xlim([1,500])
ylim([0,400])   
title('Number of spurious states')
xlabel('number of iterations')
ylabel('noise level')