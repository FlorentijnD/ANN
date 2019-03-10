NotSeperableData = [ -0.2 -0.4 +0.2 +0 -0.2; ...
      -0.2 +0.3 -0.2 +0.3 0];
NotSeperableLabels = [1 1 0 1 0];
plotpv(NotSeperableData,NotSeperableLabels);

net_NonSep = perceptron;
net_NonSep = configure(net_NonSep,NotSeperableData,NotSeperableLabels);

%%
hold on
plotpv(NotSeperableData,NotSeperableLabels);
linehandle = plotpc(net_NonSep.IW{1},net_NonSep.b{1});
%%
for a = 1:25
   [net_NonSep,Y,E] = adapt(net_NonSep,NotSeperableData,NotSeperableLabels);
   linehandle = plotpc(net_NonSep.IW{1},net_NonSep.b{1},linehandle);  drawnow;
end;

displayEndOfDemoMessage(mfilename)
