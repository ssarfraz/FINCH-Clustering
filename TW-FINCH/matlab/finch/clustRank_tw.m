function [A, orig_dist,min_sim]= clustRank_tw(mat, initial_rank, dist)

if nargin < 3
    dist = 'cosine'; % Default    
end
s=size(mat,1);  % to handle direct input of initial_rank and avoid computing pdist.. if computing 1-neigbours indices via flann
loc=mat(:,end);
mat=mat(:,1:end-1);

if ~isempty(initial_rank)
        orig_dist=[];
        min_sim=inf;
elseif  s<=90000     %% change here if you want to switch to flann on less than 90k samples
 orig_dist = pdist2(mat,mat,dist);  
 
 loc_dist= pdist2(loc,loc, 'euclidean'); % 
 
 orig_dist=orig_dist .* loc_dist;  
 
 orig_dist(logical(speye(size(orig_dist))))=inf;
 [d,initial_rank]=min(orig_dist,[],2);
 
 min_sim=double(max(d));
else
 fprintf('finding exact neghbours via pdist is not fesable on ram with data size of %d points.\nUsing flann to compute 1st-neighbours at this step ...\n\n ',s)
        [initial_rank,d] = flann_nn(loc,8);  %loc
        disp('step flann done...')      
        min_sim=sort(d);
        min_sim=median(min_sim(1:floor(length(d)*0.1)));
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