
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
    * use_ann_above_samples: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. set this for data where exact distances are not feasible to compute. [default = 70000]
    * verbos : for printing some output

Output:

* c: N x P array,  each column vector contains cluster labels for each partition P
* num_clust: shows total number of cluster in each partition P
* req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.


**Example:** Cluster the STL-10 data (13000 images of 10 object classes. We provide the used 2048 CNN resnet features.
We can load the  data from /data/STL_10/data.mat. This has 13000 vectors stored as a matrix of size (13000,2048), each vector is 2048 dimensional.

See below the notebook for an example on clustering the STL-10 data, which depicts the usage of input params as well.

- [Clustering STL-10 dataset with FINCH](notebooks/Clustering_with_FINCH.ipynb)



