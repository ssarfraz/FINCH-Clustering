
function [A, orig_dist,min_sim]= clustRank(mat, initial_rank)
% Implements the clustering eqaution
% copyright M. Saquib Sarfraz (KIT), 2019

s=size(mat,1);  % to handle direct input of initial_rank and avoid computing pdist.. if computing 1-neigbours indices via flann

if ~isempty(initial_rank)
        orig_dist=[]; min_sim=inf;
elseif  s<=70000      %% change here if you want to switch to flann on less than 70k samples
 orig_dist = pdist2(mat,mat,'cosine');  %% use euclidean here for toy 2D data where you have xy coordinates of each point to cluster
 orig_dist(logical(speye(size(orig_dist))))=inf;
 [d,initial_rank]=min(orig_dist,[],2);
 min_sim=max(d);
 
else
 fprintf('finding exact neghbours via pdist is not fesable on ram with data size of %d points.\nUsing flann to compute 1st-neighbours at this step ...\n\n ',s)
        [initial_rank,d] = flann_nn(mat,8);
        disp('step flann done...')
        min_sim=max(d);
        orig_dist=[];
end


%%% Implementation of The clustering Equation %%  

%%% Note only needs integer indices of first neigbours to directly deliver the adj matrix which has
%%% the clusters.
 
  A=sparse([1:s],initial_rank,1,s,s);
  
  A= A + sparse([1:s],[1:s],1,s,s);
  A= (A*A');
  A(logical(speye(size(A))))=0; 
  A=spones(A);
%%%
   
end
