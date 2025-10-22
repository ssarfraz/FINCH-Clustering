
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
c, num_clust, req_c = FINCH(data,
                            initial_rank=None,
                            req_clust=None,
                            distance='cosine',
                            ensure_early_exit=True,
                            ann_threshold=20_000,
                            faiss_threshold=5_000_000,
                            faiss_use_gpu= False,
                            ram_size_in_gb=None,
                            verbose=True)
```

Input:

* data: numpy array (feature vectors in rows)
* [OPTIONAL]
    * req_clust: specify required number of clusters, if set finch additionally returns the required clusters' labels
    * distance: One of sklearn's distance metrics. Recommended: 'cosine (default)' and 'euclidean'
    * initial_rank: Nx1 vector of 1-neighbour indices, optional if provided skip the first distance compute and directly used these in build.
    * ensure_early_exit: (default: True) if set it may help for Unbalanced or small datasets, ensure purity of merges and helps early exit
    * ann_threshold: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. set this for data where exact distances are not feasible to compute. [default = 20000]
    * faiss_threshold: if faiss is installed use faiss for ann above this many data samples, faster and memory efficient
    * faiss_use_gpu: run faiss on gpu if set
    * ram_size_in_gb: if provided optimized faiss kwargs are used as per data size and available RAM 
    * verbos : print some intermediate info

Output:

* c: N x P array,  each column vector contains cluster labels for each partition P
* num_clust: shows total number of cluster in each partition P
* req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.


**Example:** Cluster the STL-10 data (13000 images of 10 object classes. We provide the used 2048 CNN resnet features.
We can load the  data from /data/STL_10/data.mat. This has 13000 vectors stored as a matrix of size (13000,2048), each vector is 2048 dimensional.

See below the notebook for an example on clustering the STL-10 data, which depicts the usage of input params as well.

- [Clustering STL-10 dataset with FINCH](https://github.com/ssarfraz/FINCH-Clustering/blob/master/notebooks/Clustering_with_FINCH.ipynb)



