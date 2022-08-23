# First Integer Neighbor Clustering Hierarchy (FINCH) Algorithm

The repository contains our Python and Matlab code for the proposed FINCH clustering algorithm described in our **Efficient Parameter-free Clustering Using First Neighbor Relations** CVPR 2019 oral [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.pdf).

```
@inproceedings{finch,
    author    = {M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen}, 
    title     = {Efficient Parameter-free Clustering Using First Neighbor Relations}, 
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {8934--8943}
    year  = {2019}
}
```


## Installation
The project is available in PyPI. To install run:

`pip install finch-clust`  

**Optional**.  Install [PyNNDescent](https://github.com/lmcinnes/pynndescent) to get first neighbours for large data

To install finch with pynndescent run:

`pip install "finch-clust[ann]"`  


## Usage:

typically you would run: 
``` 
from finch import FINCH
c, num_clust, req_c = FINCH(data)

```
You can set options e.g., required number of cluster or distance etc,

```
c, num_clust, req_c = FINCH(data, initial_rank=None, req_clust=None, distance='cosine', verbose=True)
```

Input:

* data: numpy array (feature vectors in rows)
* [OPTIONAL]
    * req_c: specify required number of cluster
    * distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended: 'cosine (default)' or 'euclidean (for 2D data)'
    * initial_rank: Nx1 vector of 1-neighbour indices
    * ensure_early_exit: (default: True) if set it may help for Unbalanced or large datasets, ensure purity of merges and helps early exit
    * verbos : for printing some output

Output:

* c: N x P array,  each column vector contains cluster labels for each partition P
* num_clust: shows total number of cluster in each partition P
* req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.


**Matlab usage**

For usage in Matlab check README in the [matlab directory](matlab/README.md).

**The code and FINCH algorithm is not meant for commercial use. Please contact the author below for licensing information.**

M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
