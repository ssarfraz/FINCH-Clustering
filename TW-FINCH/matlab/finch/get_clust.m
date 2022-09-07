function [u]= get_clust(A, orig_dist,min_sim)
% get the conected components of affinity matrix

% 
 if min_sim~=inf
 ind=find((orig_dist.*A)> min_sim) ;
 A(ind)=0; 
 end
 
 
 G_d=digraph(A, 'OmitSelfLoops');
 
 u = (conncomp(G_d,'Type','weak'));
 
end