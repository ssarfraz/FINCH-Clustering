import math
import numpy as np


# -------------------- FAISS IVF-PQ (approximate) 1-NN --------------------

def _estimate_hnsw_bytes(n: int, d: int, M: int = 16) -> int:
    # float32 vectors + ~8 bytes/link (upper bound)
    return n * (4 * d + 8 * M)


def _estimate_ivfpq_bytes(n: int, d: int, nlist: int, m: int, nbits: int, id_bytes: int = 4) -> int:
    code_bytes = (m * nbits) // 8  # bytes per vector
    invlist_bytes = n * (code_bytes + id_bytes)
    coarse_centroids = nlist * d * 4  # float32
    return invlist_bytes + coarse_centroids


def _default_faiss_kwargs(n: int, metric: str, d: int = 1024, *, ram_gb: int = None) -> dict:
    """
    Choose method + params based on dataset size and RAM.
    - If ram_gb is None, assume: 64 GB for n<5M, 128 GB for 5–20M, 256 GB otherwise.
    Returns only kwargs that faiss_top1(...) accepts.
    """
    if ram_gb is None:
        ram_gb = 64 if n < 5_000_000 else (128 if n < 20_000_000 else 256)

    # --- Candidate settings
    # HNSW (CPU-only)
    h_M = 16
    h_efC = 200 if n < 2_000_000 else (300 if n < 20_000_000 else 400)
    h_efS = 64

    # IVF-PQ
    nlist_big = int(max(16_384, min(65_536, n // 256)))
    nlist_small = int(max(1_024, min(16_384, 2 * int(math.sqrt(n)))))
    big = (n >= 10_000_000)

    if big:
        nlist = nlist_big
        # Prefer 32B codes until ~30M, then 16B to save memory
        m, nbits = (32, 8) if n < 30_000_000 else (16, 8)
        nprobe = 96
        train_size = int(min(n, max(400_000, min(1_000_000, 256 * nlist))))
        topk_refine = 16
    else:
        nlist = nlist_small
        m, nbits = 16, 6  # lighter training for dev laptops
        nprobe = max(32, min(128, nlist // 16))
        train_size = min(50_000, n)
        topk_refine = 8

    # --- Method selection by memory + scale
    ram_bytes = int(ram_gb * (1 << 30))
    hnsw_bytes = _estimate_hnsw_bytes(n, d, h_M)
    ivfpq_bytes = _estimate_ivfpq_bytes(n, d, nlist, m, nbits)

    # Don’t use more than ~60% of RAM for the index; avoid HNSW beyond ~20M
    hnsw_ok = hnsw_bytes <= 0.6 * ram_bytes
    prefer_hnsw = hnsw_ok and (n < 20_000_000)

    if prefer_hnsw:
        return dict(
            method="hnsw",
            # HNSW knobs
            hnsw_M=h_M, hnsw_efC=h_efC, hnsw_efS=h_efS,
            # common
            topk_refine=8, batch_q=4096
        )
    else:
        return dict(
            method="ivfpq",
            nlist=nlist, nprobe=nprobe, m=m, nbits=nbits,
            train_size=train_size, topk_refine=topk_refine, batch_q=4096
        )


def faiss_top1(
        X: np.ndarray,
        metric: str = "cosine",
        *,
        method: str = "ivfpq",  # "ivfpq" (scales) or "hnsw" (fast CPU, more RAM)
        use_gpu: bool = False,  # IVF-PQ can be GPU-accelerated on Linux; HNSW is CPU-only
        gpu_device: int = 0,

        # IVF/HNSW knobs (supply via _default_faiss_kwargs)
        nlist: int = None,
        nprobe: int = None,
        m: int = 16,
        nbits: int = 8,
        train_size: int = None,

        # HNSW knobs
        hnsw_M: int = 16,
        hnsw_efC: int = 200,
        hnsw_efS: int = 64,

        # common
        topk_refine: int = 8,
        batch_q: int = 4096,
        verbose: bool = False,
):
    """
    Compute 1-NN indices and distances for all rows in X.
    Returns:
      nn_idx:  (n,) int32
      nn_dist: (n,) float32   # cosine distance (1 - sim) or Euclidean distance
    """
    try:
        import faiss, os
    except Exception as e:
        raise ImportError("faiss-cpu is required") from e
    '''
    # Threading hygiene (helps a lot on macOS)
    if "VECLIB_MAXIMUM_THREADS" not in os.environ:
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        faiss.omp_set_num_threads(max(1, faiss.omp_get_max_threads() // 2))
    except Exception:
        pass
    '''
    faiss.verbose = bool(verbose)

    # X = np.asarray(X, dtype=np.float32, order="C")
    n, d = X.shape
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    # Metric prep
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        Xn = X / norms
        faiss_metric = faiss.METRIC_INNER_PRODUCT  # IP on normalized vectors
        need_cosine = True
    elif metric == "euclidean":
        Xn = X
        faiss_metric = faiss.METRIC_L2
        need_cosine = False
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    method = method.lower()

    # ---------------- HNSW (CPU-only)
    if method == "hnsw":
        if use_gpu and verbose:
            print("[faiss_top1] HNSW is CPU-only; ignoring use_gpu=True")
        index = faiss.IndexHNSWFlat(d, hnsw_M)
        index.hnsw.efConstruction = int(hnsw_efC)
        index.hnsw.efSearch = int(hnsw_efS)
        if verbose:
            print(f"[faiss_top1] HNSW: M={hnsw_M}, efC={hnsw_efC}, efS={hnsw_efS}, n={n}, d={d}")

        index.add(Xn)
        if n == 1:
            return np.array([-1], dtype=np.int32), np.array([np.inf], dtype=np.float32)

        # Ask for 2 neighbors (self + NN), then drop self
        D, I = index.search(Xn, 2)

        nn_idx = I[:, 1].astype(np.int32)

        if need_cosine:
            # For normalized vectors, faiss L2 distance is squared: dist2 = 2 - 2*dot
            dist2 = D[:, 1].astype(np.float32)
            nn_dist = (dist2 / 2.0).astype(np.float32)  # cosine distance = 1 - dot = dist2/2
        else:
            # Faiss returns squared L2 for L2 space
            nn_dist = np.sqrt(D[:, 1].astype(np.float32), dtype=np.float32)
        return nn_idx, nn_dist

    # ---------------- IVF-PQ (scales, memory efficient)

    if nlist is None:
        nlist = int(max(1024, min(65_536, 2 * int(math.sqrt(n)))))
    if nprobe is None:
        nprobe = max(32, min(128, nlist // 16))
    if train_size is None:
        train_size = min(n, max(50_000, min(1_000_000, 256 * nlist)))

    if verbose:
        print(f"[faiss_top1] IVF-PQ: nlist={nlist}, nprobe={nprobe}, m={m}, nbits={nbits}, "
              f"train={min(n, train_size)} (n={n}, d={d})")

    q = faiss.IndexFlatIP(d) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(q, d, nlist, m, nbits, faiss_metric)

    if use_gpu:
        # Only available on Linux with faiss-gpu
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_device, index)

    # Train (on a subset)
    rng = np.random.default_rng(12345)
    train_sel = rng.choice(n, size=min(n, train_size), replace=False)
    index.train(Xn[train_sel])

    # Add & search with refine
    index.add(Xn)
    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)

    K = max(1, int(topk_refine))
    nn_idx = np.full(n, -1, dtype=np.int32)
    nn_score = np.full(n, (-np.inf if need_cosine else np.inf), dtype=np.float32)

    if not need_cosine:
        sq_norms = np.sum(Xn * Xn, axis=1, dtype=np.float32)

    for q0 in range(0, n, batch_q):
        q1 = min(q0 + batch_q, n)
        Q = Xn[q0:q1]
        D, I = index.search(Q, K)

        for t in range(q0, q1):
            row = t - q0
            cand = I[row]
            mask = cand >= 0
            if not np.any(mask):
                continue
            idxs = cand[mask]

            if need_cosine:
                sims = (Q[row][None, :] @ Xn[idxs].T).astype(np.float32).ravel()
                # exclude self
                sims[idxs == t] = -np.inf
                kbest = int(np.argmax(sims))
                nn_idx[t] = int(idxs[kbest])
                nn_score[t] = sims[kbest]
            else:
                dots = (Q[row][None, :] @ Xn[idxs].T).astype(np.float32).ravel()
                dist2 = sq_norms[t] + sq_norms[idxs] - 2.0 * dots
                np.maximum(dist2, 0.0, out=dist2)
                dist2[idxs == t] = np.inf
                kbest = int(np.argmin(dist2))
                nn_idx[t] = int(idxs[kbest])
                nn_score[t] = dist2[kbest]

    nn_dist = (1.0 - nn_score).astype(np.float32) if need_cosine else np.sqrt(nn_score, dtype=np.float32)
    return nn_idx, nn_dist


# --------------Optional(not used currently)---- chunked exact pairwise-----------------
def one_nn_exact_chunked(mat, metric="cosine", chunk_size=4000):
    """
    Exact 1-NN with O(N * chunk_size) memory via chunked pairwise distances.
    Returns:
        nn_idx:  (N,) int32
        nn_dist: (N,) float32
    """
    from sklearn.metrics import pairwise_distances_chunked
    N = mat.shape[0]
    nn_idx = np.empty(N, dtype=np.int32)
    nn_dist = np.empty(N, dtype=np.float32)

    start = 0
    for block in pairwise_distances_chunked(
            mat, mat, metric=metric, n_jobs=-1, working_memory=None, reduce_func=None
    ):
        q = block.shape[0]
        row = np.arange(q)
        # mask only the diagonal entries of this (query, reference) block
        block[row, start + row] = np.inf
        # per-row argmin
        j_min = np.argmin(block, axis=1)
        d_min = block[row, j_min]
        nn_idx[start:start + q] = j_min.astype(np.int32, copy=False)
        nn_dist[start:start + q] = d_min.astype(np.float32, copy=False)
        start += q

    return nn_idx, nn_dist
