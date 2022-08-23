
##### Python Requirements

See requirements.txt

[Optional]: pip install pynndescent to get first neighbours for large data


## Usage:
typically you would run: 
``` 
from finch import FINCH
c, num_clust, req_c = FINCH(data)

```
You can set options e.g., required number of cluster [optional] or distance etc,

```
c, num_clust, req_c = FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=False, verbose=True)
```

Input:

* data: numpy array (feature vectors in rows)
* [OPTIONAL]
    * req_c: specify required number of cluster
    * distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended: 'cosine (default)' and 'euclidean (for 2D toy data)'
    * initial_rank: Nx1 vector of 1-neighbour indices
    * ensure_early_exit: (default: True) if set it may help for Unbalanced or large datasets, ensure purity of merges and helps early exit
    * verbos : for printing some output

Output:

* c: N x P array,  each column vector contains cluster labels for each partition P
* num_clust: shows total number of cluster in each partition P
* req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.


**Example:** Cluster the STL-10 data (13000 images of 10 object classes. We provide the used 2048 CNN resnet features.
Please load the  data from /data/STL_10/data.mat. This has 13000 vectors stored as a matrix of size (13000,2048), each vector is 2048 dimensional.


Now cluster it using FINCH, run : 

```
c, num_clust, req_c = FINCH(data, req_clust=None, distance='cosine', verbose=True)
```


it returns the cluster labels for each partition in the variable c which will be of size (N x numPartitions) e.g., (13000, 5) in this case. Each column in array c provides cluster labels for that partition.

num_clust provides how many cluster it has produced in each partition or step of the run.


As you see: num_clust = [2061, 177, 37, 10, 2]

inidicating it found 2061 clusters in step 1, 177 in step 2 and so on to 10 clusters in step 4. You can pick the respective cluster labels for the data in the returned array c. For example, c(:,4) will provide labels for 10 clustering result.

### Required number of Clusters
if you require a certain number od cluster, just set the req_clust argument

``````
c, num_clust, req_c = FINCH(data, req_clust=15)
``````

 Provides the required 15 clusters by refining the respective FINCH partition (37 clusters), from FINCH returned c mat in the above example. See function help for more details.

