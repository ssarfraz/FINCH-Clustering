function [c, num_clust]= FINCH(data, initial_rank, tw_finch, verbose, dist)
% Input
% data: feature Matrix (feature vecotrs in rows)
% initial_rank [Optional]: Nx1  1-neighbour indices vector ... pass empty [] to compute the 1st neighbor via pdist or flann
% tw_finch : true
% verbos : true for printing some output

% Output
% c: N x P matrix  Each coloumn vector contains cluster labels for each partion P
% num:clust: shows total number of cluster in each partion P
%
% The code implements the FINCH algorithm described in our CVPR 2019 paper
% M. Saquib Sarfraz, Vivek Sharma and Ranier Stiefelhagen,"Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR 2019.
% https://arxiv.org/abs/1902.11266
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% For academic purpose only. The code or its re-implemntation should not be used for commercial use.
% Please contact the author below for licensing information.
% Copyright
% M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
% Karlsruhe Institute of Technology (KIT)



%% Initialize FINCH clustering
if nargin < 5
    dist = 'cosine'; % Default    
end

min_sim = inf;
clustRank_fn= @clustRank;
  

if tw_finch
      n_frames = size(data, 1);
      time_index=(1:n_frames)/n_frames;
      data = [data, time_index'];      
      clustRank_fn=@clustRank_tw; 
end
%%     
   [Affinity_,  orig_dist, ~]= clustRank_fn(data, initial_rank, dist);
    
   initial_rank=[];
  
  [Group_] = get_clust(Affinity_, [],inf);
    
  [c,num_clust, mat]=get_merge([],Group_,data);
    
  if verbose
  fprintf('Partition 1 : %d clusters\n',num_clust)
  end
  
%% for sanity check, early stoping when using pdist...%% can be commented out
%       if ~isempty(orig_dist)
%        min_sim=  double(max(orig_dist(Affinity_>0)));
%     end
    %%    
    
    exit_clust=inf;
    c_=c;
k=2;
while exit_clust>1    
    [Affinity_,  orig_dist,~]= clustRank_fn(mat,initial_rank, dist);  
      
    [u] = get_clust(Affinity_, double(orig_dist),min_sim);    
    [c_,num_clust_curr, mat]=get_merge(c_, u, data);     
    
    
   
    
    num_clust =[num_clust, num_clust_curr]; 
    c = [c, c_];    
      
  % exit if cluster is 1
   exit_clust=num_clust(end-1)-num_clust_curr;  
     % 
     if num_clust_curr==1 | exit_clust<1
         num_clust=num_clust(1:end-1);
         c=c(:,1:end-1);
         exit_clust=0;
        break
     end
    if verbose                
      fprintf('Partition %d : %d clusters\n',k,num_clust(k))
    end 
    k=k+1;
    
end

end









