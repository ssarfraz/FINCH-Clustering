from typing import Optional
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from finch.finch_utils import faiss_top1, _default_faiss_kwargs
import warnings

try:
    from pynndescent import NNDescent
    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent is not installed: {}'.format(e))
    pynndescent_available = False
    pass

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def clust_rank(
        mat,
        initial_rank=None,
        metric="cosine",
        verbose=False,
        ann_threshold=10_000,
        random_state=None,
        *,
        # FAISS branch (very large)
        use_faiss=True,
        faiss_threshold=5_000_000,
        faiss_use_gpu=False,
        ram_size_in_gb=None,
        faiss_kwargs=None,  # dict passed to _faiss_top1_ivfpq
):
    """
    Compute 1-NN for FINCH:

      n = mat.shape[0]
      • if initial_rank is provided: use it (legacy).
      • elif n <= ann_threshold; exact 1-NN via pairwise.
      • elif use_faiss and n >= faiss_threshold and metric in {"cosine","euclidean"}:
           - FAISS IVF-PQ or HSNW top-1
      • else:
           - NN-Descent with k=2 (memory-friendly, fast mid-range).

    Returns
    -------
    sparse_adjacency_matrix : csr_matrix (n, n)
        One outgoing edge per row (i -> 1-NN(i)).
    orig_dist : ndarray
        If pairwise exact path: (n, n) dense distances (diag set large).
        Else: shape (n, 2): [:,0]=large sentinel (1e12), [:,1]=1-NN distance.
    initial_rank : ndarray (n,)
        Index of the 1st non-self neighbor for each point.
    knn_index : object or None
        NNDescent index if used; else None.
    """

    s = mat.shape[0]
    knn_index = None

    if faiss_kwargs is None:
        faiss_kwargs = _default_faiss_kwargs(n=s, metric=metric, d=mat.shape[1], ram_gb=ram_size_in_gb)

    else:
        # merge user overrides onto auto defaults
        auto = _default_faiss_kwargs(s, metric)
        auto.update(faiss_kwargs)
        faiss_kwargs = auto

    # --- Legacy: user-provided 1-NN indices ---
    if initial_rank is not None:
        # Minimal placeholder to keep downstream shape checks happy
        orig_dist = np.empty((1, 1), dtype=np.float32)

    else:
        # ---------- Exact regime (n <= ann_threshold) ----------
        if s <= ann_threshold:
            orig_dist = metrics.pairwise_distances(mat, mat, metric=metric).astype(np.float32, copy=False)
            np.fill_diagonal(orig_dist, np.float32(1e12))
            initial_rank = np.argmin(orig_dist, axis=1).astype(np.int32, copy=False)

        # ---------- Very large: FAISS IVF-PQ (optional) ----------
        elif use_faiss and (s >= faiss_threshold) and (metric in {"cosine", "euclidean"}):
            if verbose:
                print(f"[FINCH] FAISS 1-NN (n={s}, metric='{metric}')")
            try:
                nn_idx, nn_dst = faiss_top1(mat, metric=metric, use_gpu=faiss_use_gpu, verbose=verbose, **faiss_kwargs)
                orig_dist = np.empty((s, 2), dtype=np.float32)
                orig_dist[:, 0] = np.float32(1e12)
                orig_dist[:, 1] = nn_dst.astype(np.float32, copy=False)
                initial_rank = nn_idx.astype(np.int32, copy=False)
            except Exception as e:
                if verbose:
                    print(f"[FINCH] FAISS path failed ({e}); falling back to NN-Descent.")
                if not pynndescent_available:
                    raise MemoryError(
                        "You should install either faiss or pynndescent for inputs larger than ann_threshold: {} samples or set ann_threshold = data_size if your machine can handle exact pairwsie distance compute.".format(
                            ann_threshold))
                knn_index = NNDescent(
                    mat, n_neighbors=2, metric=metric,
                    verbose=verbose, random_state=random_state
                )
                result, dist = knn_index.neighbor_graph
                initial_rank = result[:, 1].astype(np.int32, copy=False)
                orig_dist = dist.astype(np.float32, copy=False)
                orig_dist[:, 0] = np.float32(1e12)

        # ---------- Mid / large (default): NN-Descent ----------
        else:
            if not pynndescent_available:
                raise MemoryError(
                    "You should install either faiss or pynndescent for inputs larger than ann_threshold: {} samples or set ann_threshold = data_size if your machine can handle exact pairwsie distance compute.".format(
                        ann_threshold))
            if verbose:
                print(f"[FINCH] PyNNDescent (k=2, n={s}, metric='{metric}')")
            knn_index = NNDescent(
                mat, n_neighbors=2, metric=metric,
                verbose=verbose, random_state=random_state
            )
            result, dist = knn_index.neighbor_graph
            initial_rank = result[:, 1].astype(np.int32, copy=False)
            orig_dist = dist.astype(np.float32, copy=False)
            orig_dist[:, 0] = np.float32(1e12)

    # Build sparse 1-NN adjacency (one outgoing edge per row)
    rows = np.arange(s, dtype=np.int32)
    cols = initial_rank
    data = np.ones_like(rows, dtype=np.float32)
    sparse_adjacency_matrix = sp.csr_matrix((data, (rows, cols)), shape=(s, s))

    return sparse_adjacency_matrix, orig_dist, initial_rank, knn_index


def cool_mean(data: np.ndarray, u: np.ndarray):
    """Efficiently calculate the mean of all rows of a matrix M over a partition u. The number of classes in the
   partition is implicitly defined from the values of u.


   Parameters
   ----------
       data: Matrix of dimensions (n, f) with n data points of f features each.

       u: Partition of the data points in the form of an (n, ) array with k different integer values.

   Returns
   -------
       group_mean: A (k, f) matrix with the vectors averaged over the k partition values.
    """

    s = data.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ data) / nf[:, None]


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(
        csgraph=a, directed=True, connection="weak", return_labels=True
    )
    return u, num_clust


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)

    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, use_ann_above_samples, verbose):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist, _, _ = clust_rank(mat, initial_rank=None, ann_threshold=use_ann_above_samples, metric=distance,
                                          verbose=verbose)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


# noinspection PyPep8Naming
def FINCH(
        data: np.ndarray,
        req_clust: Optional[int] = None,
        initial_rank: Optional[np.ndarray] = None,
        distance: str = "cosine",
        ensure_early_exit: bool = True,
        ann_threshold: int = 20_000,
        faiss_threshold: int = 5_000_000,
        faiss_use_gpu: bool = False,
        ram_size_in_gb: Optional[int] = None,
        faiss_kwargs: dict = None,
        return_meta_labels: bool = False,
        verbose: bool = False,
        random_state: Optional[int] = None,
):
    """FINCH clustering algorithm.

    Parameters
    ----------
        data: array, shape (n_samples, n_features)
            Input matrix with features in rows.

        req_clust: Set output number of clusters (optional).

        initial_rank: array, shape (n_samples, 1) (optional)
            First integer neighbor indices.

        distance: str (default 'cosine')
            One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.

        ensure_early_exit: bool (default True)
            May help in large, high dim datasets, ensure purity of merges and helps early exit.

        verbose: bool (default True)
            Print verbose output.

        ann_threshold: int (default 20_000)
            Data size threshold above which nearest neighbors are approximated with ANNs.

        faiss_threshold: int (default 5_000_000)
            if ANN method FAISS is available , will use it above this number of samples for 1-NN computations, else will use NNDescent

        faiss_use_gpu: bool (default False)
            if faiss-gpu is installed run faiss on gpu for faster compute

        faiss_kwargs: Optional (default None, sets automatically)
            faiss kwargs

        ram_size_in_gb: Optional[int] (default None)
            size of available RAM in GB, if provided use it in estimating optimized faiss params

        return_meta_labels:  bool (default False)
            additonally returns  finch level 0 cluster means  and  per level mapping labels

        random_state: Optional[int] (default None)
            An optional random state for reproducibility purposes. It fixes the state of ANN.

    Returns
    -------
        c: array of shape (n_samples, n_partitions)
            Matrix with labels indicating cluster participation. There is one column per partition.

        num_clust: array of shape (n_partitions)
            Number of clusters per partition.

        requested_c: array of shape (n_samples, 1)
            Labels of the requested clusters if req_c is set else None

        [Optional returns]
        partition_clustering: list of arrays of shapes equal to the values of num_clust
            List of arrays with labels indicating the centroids cluster participation per level.

        lowest_level_centroids: array of shape (num_clust[0], n_features)
            The feature coordinates of the lowest level centroids.

    References
    ----------
        The code implements the FINCH algorithm described in our CVPR 2019 paper
        [1] Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.pdf
        Original code author:
            M. Saquib Sarfraz (saquibsarfraz@gmail.com)
    """
    data = np.asarray(data, dtype=np.float32, order="C")

    min_sim = None

    adj, orig_dist, first_neighbors, _ = clust_rank(
        data,
        initial_rank,
        distance,
        verbose=verbose,
        ann_threshold=ann_threshold,
        random_state=random_state,
        faiss_threshold=faiss_threshold,
        faiss_use_gpu=faiss_use_gpu,
        ram_size_in_gb=ram_size_in_gb,
        faiss_kwargs=faiss_kwargs,  # sets auto if None
    )

    initial_rank = None

    group, num_clust = get_clust(adj, [], min_sim)

    c, mat = get_merge([], group, data)

    if return_meta_labels:
        lowest_level_centroids = mat

    if verbose:
        print("Level 0: {} clusters".format(num_clust))

    if ensure_early_exit:
        if orig_dist.ndim == 2 and orig_dist.shape[-1] > 2:
            min_sim = float(np.max(orig_dist * adj.toarray()))

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]
    partition_clustering = []

    while exit_clust > 1:
        adj, orig_dist, first_neighbors, knn_index = clust_rank(
            mat,
            initial_rank,
            distance,
            verbose=verbose,
            ann_threshold=ann_threshold,
            random_state=random_state,
            faiss_threshold=faiss_threshold,
            faiss_use_gpu=faiss_use_gpu,
            ram_size_in_gb=ram_size_in_gb,
            faiss_kwargs=faiss_kwargs,  # sets auto if None
        )

        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)

        if return_meta_labels:
            partition_clustering.append(u)

        c_, mat = get_merge(c_, u, data)
        c = np.column_stack((c, c_))

        num_clust.append(num_clust_curr)
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust <= 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print("Level {}: {} clusters".format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            if req_clust > num_clust[0]:
                print(f'requested number of clusters are larger than FINCH first partition with {num_clust[0]} clusters . Returning {num_clust[0]} clusters')
                requested_c = c[:, 0]
            else:
                ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
                requested_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, ann_threshold, verbose)

    else:
        requested_c = None

    if return_meta_labels:
        return c, num_clust, requested_c, partition_clustering, lowest_level_centroids
    else:
        return c, num_clust, requested_c
