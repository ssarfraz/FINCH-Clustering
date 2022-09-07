function req_c= req_numclust(c, data,req_clust, tw_finch, dist)
%% Algo 2 procedure to refine a partion for required number of clusters
% part of FINCH code
% copyright M. Saquib Sarfraz (KIT), 2019
%
%Input:
% c: partition to refine ..  one of the partions returned form FINCH
% Example: 
% Suppose FINCH returns partitions with clusters:
% num_clust=[2061,177,37,10,2] and we want 15 clusters
% We should then pass the closest larger partion to 15 i.e. 3rd partition with 37 cluster in this example.
% c should be then c(:,3) to refine the 3rd FINCH partion from FINCH returne c mat
% we will run : req_c= req_numclust(c(:,3), data, 15)
%
% data: data mat as used in FINCH (N x d) feature vecotrs in rows
% req_clust: required number of clusters
% Output:
% req_c : required number of clusters: 1xN vector of  cluster labels
if nargin < 5
    dist = 'cosine'; % Default    
end

data = single(data);

if tw_finch
    n_frames = size(data, 1);
    time_index=(1:n_frames)/n_frames;
    data = [data, time_index']; 
    clustRank_fn=@clustRank_tw; 
else
    clustRank_fn=@clustRank; 
end

       iter=length(unique(c))- req_clust;

       [c_,~, mat]=get_merge([],c',data); 
    for i=1:iter

    [Affinity_,  orig_dist]= clustRank_fn(mat,[]);
    
    Affinity_= top_affinity(Affinity_, orig_dist); %update affinity only keeping one merge at a time
    
    [u] = get_clust(Affinity_, [],inf);
    
    
   [c_,~, mat]=get_merge(c_,u,data);    
    
  
    end
    
    %req_clust_final =length(unique(c_));
    req_c = c_;
    
  function Affinity_= top_affinity(Affinity_, orig_dist)
    
        Affinity_(logical(speye(size(Affinity_))))=0;
       
        in_=find(Affinity_>0);
        [~,v]=sort(orig_dist(in_)); idx=in_(v(1:2)); 
        Affinity_= zeros(size(Affinity_),'logical');
        Affinity_(idx)=1;
             
    end    
    
    end