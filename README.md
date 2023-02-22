# First Integer Neighbor Clustering Hierarchy (FINCH) Algorithm
![alt text](https://raw.githubusercontent.com/ssarfraz/FINCH-Clustering/master/data/toy_data_vis.jpg)

FINCH is a parameter-free fast and scalable clustering algorithm. it stands out for its speed and clustering quality.
 The algorithm is described in our paper **Efficient Parameter-free Clustering Using First Neighbor Relations** published in CVPR 2019 . [Read Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.pdf).


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

For more details on meaning of input arguments check README in [finch directory](finch/README.md). 

**Matlab usage**

Correponding Matlab implementation is provided in the [matlab directory](https://github.com/ssarfraz/FINCH-Clustering/tree/master/matlab/README.md).

## Demos

The following demo notebooks are available to see the usage in clustering a dataset.

1. [Basic usage on 2D toy data](https://github.com/ssarfraz/FINCH-Clustering/blob/master/notebooks/Basic_usage.ipynb)
2. [Clustering STL-10 dataset with FINCH](https://github.com/ssarfraz/FINCH-Clustering/blob/master/notebooks/Clustering_with_FINCH.ipynb)
 






## Relevant tools built on FINCH
- [h-nne](https://github.com/koulakis/h-nne): See also our [h-nne](https://github.com/koulakis/h-nne) method which uses FINCH for fast dimenionality reduction  and visualization applications.

- [TW-FINCH](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH): Also see our [TW-FINCH](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH) variant which is useful for video segmentation.
## Citation 
```
@inproceedings{finch,
    author    = {M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen}, 
    title     = {Efficient Parameter-free Clustering Using First Neighbor Relations}, 
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {8934--8943}
    year  = {2019}
}

```

**The code and FINCH algorithm is not meant for commercial use. Please contact the author for licensing information.**
