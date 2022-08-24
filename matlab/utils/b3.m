function [DATA] = b3(L,K,N_L,N_K)
% [score] = b3(Labels,Clusters)
%
% This function determines the extrinsic clustering quality b-cubed measure 
% using a set of known labels for a data set, and cluster assignmnets of 
% each data point stored as either vector or cell arrays where each
% cell/row represets a data point in which an array containing class or cluster
% assignments is stored.
%
% Inputs
% ------------------
%       L: An NxM matrix containing class labels for each data point. Each 
%          row represents the ith data point and each column contains a 0, or 1
%          in the jth column indicating membership of label j.
%          Alternatively, if hard class labels are available, L can be input 
%          as an Nx1 vector where each entry is its class label
%
%       K: Defined identically to L except for this variable stores cluster
%          assignments for each data point
%
% Outputs
%--------------------------------
% F_B3: This F-measure using the b-cubed metric
% P_B3: The b-cubed precision
% R_B3: The b-cubed recall
%
%-----------------------------------------------
% % Not our function
% Author: Matthew Wiesner
% Email : wiesner@jhu.edu 
% Institute: Johns Hopkins University Electrical and Computer Engineering
%
% Refences: DESCRIPTION OF THE UPENN CAMP SYSTEM AS USED FOR COREFERENCE,
%           Breck Baldwin, Tom Morton, Amit Bagga, Jason Baldridge, 
%           Raman Chandraseker, Alexis Dimitriadis, Kieran Snyder, 
%           Magdalena Wolska, Institute for Research in Cognitive Science
%
%           A.A. Aroch-Villarruel Pattern Recognition 6th Mexican Conference,
%           MCPR 2014 Proceedings Paper, p.115
% ----------------------------------------------
if(isempty(K) || isempty(L))
    disp('The clusters assignments or labels are empty')
    DATA = struct([]);
    return
end
N = size(L,1);

% Based on first reference
if(nargin == 2)
    [~,~,K] = unique(K);
    [~,~,L] = unique(L);
    
    % This block ensures that the function will work on legacy MATLAB
    % version where the returned values were rows instead of columns
    if(isrow(K))
        K = K';
    end
    if(isrow(L))
        L = L';
    end
    X_L = repmat(L,1,max(L(:)));
    X_K = repmat(K,1,max(K(:)));

    % Each column j has truth membership of the ith data point in Label j
    L_j = (X_L == repmat((1:max(L(:))),N,1)); 

    % Each column j has truth membership of the ith data point in Cluster j
    K_j = (X_K == repmat((1:max(K(:))),N,1));
    
    % The ijth element of the partitions matrix P_ij is the number of elements 
    % present in Label i and Cluster j
    P_ij = double(L_j)'*double(K_j);

    % The ith entry of S_i/T_i is the number of elements present in 
    % label/cluster i
    S_i = sum(P_ij,2);
    T_i = sum(P_ij,1);

    % The recall and precision for label i
    R_i = sum(P_ij.*P_ij,2)./(S_i.^2);
    P_i = sum(P_ij'.*P_ij',2)./(T_i.^2)';

    % Two different weighting schemes for average recall.
% Equal weighting of classes
% Equal weighting of entities
    
%    R_N = sum(R_i)/max(L(:));
    R_p = sum(S_i.*R_i)/N;

    % Two different weighting schemes for average precision
%    P_N = sum(P_i)/max(K(:));
    P_p = sum(T_i'.*P_i)/N;

    F_p = harmmean([P_p R_p]);
%    F_N = harmmean([P_N R_N]);

% Based on extension to multiplicity case as presented in the second
% reference (This section may not yet work, and does not scale well!!!)
else
    L_membership = cell2mat(cellfun(@(a_ci) ismember(1:N_L,a_ci),L,'UniformOutput',false));
    K_membership = cell2mat(cellfun(@(a_ci) ismember(1:N_K,a_ci),K,'UniformOutput',false));
    confL = double(L_membership)*double(L_membership)';
    confK = double(K_membership)*double(K_membership)';
    
    % Generate the multiplicity precision and recall function matrices
    M = min(confK,confL);
    MP = M;
    MR = M;
    %MP = M./confK;
    MP(confK ~= 0) = M(confK ~= 0)./confK(confK ~= 0);
    %MR = M./confL;
    MR(confL ~= 0) = M(confL ~= 0)./confL(confL ~= 0);

    % Calculate the B3 metrics
    P_p = sum(MP,2)./sum(confK ~= 0,2);
    %P_p = nanmean(MP,2);
    R_p = sum(MR,2)./sum(confL ~= 0,2);
    %R_p = nanmean(MR,2);
    F_p = harmmean([mean(P_p) mean(R_p)]);
end

DATA.P = P_p;
DATA.R = R_p;
DATA.F = F_p;

end
