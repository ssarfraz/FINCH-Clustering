%addpath('Your-flann-install-Path/flann/install/share/flann/matlab');
%please install flann as per instruction on flann website
% only needed if estimating first neigbours via kdtree for big data
function [initial_rank, dist]= flann_nn(data,tree)
%data feat vect in coloumns
flann_set_distance_type('euclidean')
% 
 [index, parameters, speedup] = flann_build_index(data', struct('algorithm','kdtree','trees',tree,'cores',0));

parameters.checks=128;
[initial_rank, dist] = flann_search(index,data',2,parameters);
initial_rank= double(initial_rank(2,:));
dist=dist(2,:);
flann_free_index(index)

return