import numpy as np
import scipy
from scipy.spatial.distance import pdist,cdist
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import networkx as nx
import pdb

def clustRank(mat, initial_rank):

    # orig_dist = pdist(mat,'cosine')
    orig_dist = cdist(mat,mat,'cosine')
    np.fill_diagonal(orig_dist, float('inf'))
    
    d = np.min(orig_dist,axis=1)
    initial_rank = np.argmin(orig_dist, axis=1)
    min_sim=np.max(d)

    data_l = orig_dist.shape[0]
    value = np.ones(data_l)

    # print (value,np.arange(data_l),initial_rank)
    coo_A = coo_matrix((value,(np.arange(data_l),initial_rank)), shape=(data_l,data_l))
    diag_A = coo_matrix((value,(np.arange(data_l),np.arange(data_l))), shape=(data_l,data_l))

    coo_A = coo_A + diag_A
    coo_A = coo_A * coo_A.T
    coo_A[np.diag_indices(data_l)] = 0
    coo_A.eliminate_zeros()

    return coo_A,orig_dist

def getClust(A,orig_dist,min_sim):
    if min_sim != float('inf'):
        print ('in')
        # ind=find((orig_dist.*A)> min_sim)
        # A(ind)=0

    # G_d = nx.from_scipy_sparse_matrix(A,create_using = nx.DiGraph())
    G_ = nx.from_scipy_sparse_matrix(A,create_using = nx.Graph())

    conn_components = nx.connected_components(G_)

    return conn_components

def mergeClust(c,group,feature):

    # ref2nf
    subClustSize =[]
    sortedIdx = []
    clustNum = 0

    for c in group:
        subClustSize.append(len(c))
        sortedIdx.extend(list(c))
        clustNum = clustNum+1    
    # print (feature)

    sortedFeature = feature[sortedIdx,:]
    featZero = np.zeros((1,2048))
    sortedFeature = np.concatenate((featZero,sortedFeature))
    subClustSize = [0] + subClustSize
    # print (subClustSize,sortedIdx)
    # print (sortedFeature)

    sumFeature = sortedFeature.cumsum(0)
    sumClustSize = np.array(subClustSize).cumsum(0)

    # for i in np.arange(len(subClustSize)-1):
    #     print (i+1,i,sumClustSize[i+1],sumClustSize[i],subClustSize[i+1])
    #     print (sumFeature[sumClustSize[i+1]] - sumFeature[sumClustSize[i]])
    #     print ((sumFeature[sumClustSize[i+1]] - sumFeature[sumClustSize[i]])/subClustSize[i+1])

    mean_feature = [ (sumFeature[sumClustSize[i+1]] - sumFeature[sumClustSize[i]])/subClustSize[i+1] for i in np.arange(len(subClustSize)-1)]

    return group,clustNum,mean_feature

def FINCH(data,initial_rank,verbose):
    min_sim=float('inf')

    Affinity_,orig_dist = clustRank(data,initial_rank)
    
    feat_dist = orig_dist
    initial_rank=[]

    Group_ = getClust(Affinity_, [], float('inf'))

    c,num_clust, mat=mergeClust([],Group_,data)

    if verbose==1:
        print('Partition 1 : {:d} clusters'.format(num_clust))

    aff_dist= orig_dist*(Affinity_.toarray())
    aff_dist = np.nan_to_num(aff_dist)
    max_d = np.max(aff_dist)
    print (max_d)

    exit_clust=float('inf')
    c_=c

    k=1
    while exit_clust>1:
        Affinity_,  orig_dist = clustRank(mat,initial_rank)
        u = getClust(Affinity_, orig_dist,min_sim)
        c_,num_clust_curr, mat=mergeClust(c_, u, data)
        print (num_clust_curr)
        num_clust =[num_clust, num_clust_curr]
        c = [c, c_]
        print (c_)
        exit_clust=num_clust[-1]-num_clust_curr
        if num_clust_curr==1 | exit_clust<1:
            # num_clust=num_clust(1:end-1)
            # c=c(:,1:end-1)
            exit_clust=0
            break
        if verbose==1:
            print('Partition {:d}: {:d} clusters'.format(k,num_clust[k]))
        k=k+1


if __name__ == '__main__':

    print ('load data')
    data = np.loadtxt('ZL_cluster_eazy100_feat_c_woREA.txt',dtype=np.float32)
    print (data.shape)
    # data = np.ones( (5,24) )
    # data = np.loadtxt('ZL_cluster_eazy100_feat_c_woREA.txt')
    FINCH(data,[],1)