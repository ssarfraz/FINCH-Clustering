# First Integer Neighbor Clustering Hierarchy (FINCH) Algorithm

The repository contains our Matlab code for the proposed FINCH clustering algorithm described in our **Efficient Parameter-free Clustering Using First Neighbor Relations** CVPR 2019 oral [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.pdf).


##### Requirements

Matlab : 2017 or above: may run on earlier versions as well

**Optional**. install [flann](https://github.com/mariusmuja/flann) to get first neighbours from Kd-tree for large data


**Matlab :** Please go in the path where you copied this folder or add its path to your Matlab path.

``` 
[c, num_clust]= FINCH(data, initial_rank, verbose);
```

Input:

* data: data Matrix (feature vecotrs in rows)
* initial_rank [Optional]: Nx1  (1-neighbour) indices vector. Pass empty [] to compute the 1st neighbor via pdist or flann
* verbos : printing some output

Output:

* c: N x P matrix  Each column vector contains cluster labels for each partition P
* num_clust: shows total number of cluster in each partition P

In Matlab typically you would run: 
```
[c, num_clust]= FINCH(data,[], 1);
```


**Example:** Cluster the STL-10 data (13000 images of 10 object classes. We provide the used 2048 CNN resnet features.
Please load the  data in Matlab from /data/STL_10/data.mat. This has 13000 vectors stored as a matrix of size (13000,2048), each vector is 2048 dimensional.


Now cluster it using FINCH, run the above command with tic toc to see the runtime. The run time includes computing first neighbours via exact distance and every thing.

![alt text](../data/screenshot.png)

On our machine it was < 3 seconds, it should be about the same depending upon if your machine has similar specs.

it returns the cluster labels for each partition in the variable c which will be of size (N x numPartitions) e.g., (13000, 5) in this case. Each column in array c provides cluster labels for that partition.

num_clust provides how many cluster it has produced in each partition or step of the run.


As you see: num_clust = [2061, 177, 37, 10, 2]

inidicating it found 2061 clusters in step 1, 177 in step 2 and so on to 10 clusters in step 4. You can pick the respective cluster labels for the data in the returned array c. For example, c(:,4) will provide labels for 10 clustering result.

**[Evaluation]**: 
The true labels for this data are also provided in the same repo in label.mat file. You can run any performance metric e.g., to compute NMI metric use nmi.mat function (provided in utils) and run nmi(labels, c(:,4)), or compute BCubed Fscore with b3(labels, c(:,4)).

Similarly you can run FINCH on other datasets e.g., we provide the used mnist10k and mice protein data.

### Required number of Clusters

We provide a very simple approximation to refine one of the FINCH partition to come down to required number of clusters. However, please note this is not recommended and provided here just for completeness.
One could use better ways to refine a partition, if needed.

```req_c = req_numclust(c, data,req_clust)```

**Example**:
```
req_c= req_numclust(c(:,3), data, 15)
```
 Provides the required 15 clusters by refining the respective FINCH partition (37 clusters), from FINCH returned c mat in the above example. See function help for more details.


### Note:
To run it on large data samples, For matlab, install flann library [flann github page](https://github.com/mariusmuja/flann) (add path to matlab, see [flann_nn.m](https://github.com/ssarfraz/FINCH-Clustering/blob/master/FINCH_Core/flann_nn.m)).
Note that you can also change when to switch to using Approximate NN  in the matlab function [clustRank.m] line 10.

**Finally**, if you use FINCH on 2D toy data (for visualization) then use euclidean distance. the file [clustRank.m] line 11. as here you have xy coordinates of each point to cluster.

