## Usage:


``` 
from finch import FINCH
c, num_clust, req_c = FINCH(data, req_clust=K, tw_finch=True)

```


Input:

* data:  feature vectors of one video [numpy array]
* req_c: specify required number of cluster, [for TW-FICNH, this is the average # of actions of the activity]
* tw_finch: boolean to run TW-FINCH if False will run FINCH. 
* [OPTIONAL]
    * distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended: 'cosine (default)' 
    * initial_rank: Nx1 vector of 1-neighbour indices    
    * verbos : for printing some output

Output:

* c: N x P array,  each column vector contains cluster labels for each partition P
* num_clust: shows total number of cluster in each partition P
* req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.. For TW-FINCH this should be set and used as cluster labels