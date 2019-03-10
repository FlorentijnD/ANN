SeperableData = [ -0.2 -0.4 +0.2 +0; ...
      -0.2 +0.3 -0.2 +0.3];
SeperableLabels = [1 1 0 1];
plotpv(SeperableData,SeperableLabels);

net_Sep = perceptron;
net_Sep = configure(net_Sep,SeperableData,SeperableLabels);

%%
hold on
plotpv(SeperableData,SeperableLabels);
linehandle = plotpc(net_Sep.IW{1},net_Sep.b{1});

%%

for a = 1:25
   [net_Sep,Y,E] = adapt(net_Sep,SeperableData,SeperableLabels);
   linehandle = plotpc(net_Sep.IW{1},net_Sep.b{1},linehandle);  drawnow;
end;

displayEndOfDemoMessage(mfilename)
