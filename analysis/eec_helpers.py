"""Shared loaders and EEC math used by every script in `code/analysis/`.

This module is the single source of truth for:

  * loading time-stamped event sequences from a dataset folder, in the
    standard `edges_list.txt` / `edges_timings.txt` schema documented in
    `data/README.md`;
  * binning those event sequences into a fixed time-window width Δ so
    that one obtains the per-edge event-count vector
    \\bar e_i(\\ell) defined in Eq. (1) of the paper;
  * computing the EEC matrix between every pair of frequent edges /
    hyperedges, either as the cosine similarity (main-text Eq. (2)) or
    as the Pearson correlation coefficient (appendix Eq. (5));
  * classifying pairs by the number of shared nodes (0, 1, or 2);
  * deciding whether an intersecting pair is "open" or "closed" — i.e.,
    whether the unshared endpoints co-occur in some static hyperedge of
    any size.

Every analysis script in this folder imports from this module rather
than rolling its own loader, so any change to the data schema needs to
be made here only.

The companion script `compile_v9.py` (Pearson) and its monkey-patch
wrapper `compile_v9_eec2.py` (cosine) call into these helpers; see
`code/analysis/README.md` for an end-to-end run.
"""

import os
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# 1. Loaders — read `edges_list.txt` and `edges_timings.txt`
# ---------------------------------------------------------------------------

def load_size_k(folder, k, threshold):
    """Load every size-k (hyper)edge that has at least `threshold` events.

    The two on-disk files have parallel structure:

      edges_list.txt    : "eid size n1 n2 [n3 ...]"      (one row per edge)
      edges_timings.txt : "eid size t1 t2 ..."           (one row per edge)

    All timestamps are expressed in the dataset's native time unit and
    have already been shifted so that the first event in the dataset is
    at t=0 (see `data/README.md`).

    Parameters
    ----------
    folder : str
        Absolute path to the dataset folder, e.g.
        `data/networks/college-msg` or `data/hypergraphs/email-Eu`.
    k : int
        Hyperedge size to retain (typically 2 or 3).
    threshold : int
        Minimum number of time-stamped events on an edge for it to be
        considered "frequent" and included in the analysis.

    Returns
    -------
    eids : list of int
        Edge identifiers as recorded on disk.
    nodes_list : list of tuple of int
        Sorted node ids of each retained edge, parallel to `eids`.
    times_list : list of np.ndarray (int64)
        Event timestamps for each retained edge, parallel to `eids`.
    """
    list_path = os.path.join(folder, "edges_list.txt")
    timing_path = os.path.join(folder, "edges_timings.txt")

    # First pass: harvest node tuples for every size-k edge.
    nodes_of = {}
    with open(list_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != k:
                continue
            eid = int(parts[0])
            nodes_of[eid] = tuple(sorted(int(x) for x in parts[2:2 + sz]))

    # Second pass: harvest timestamps, retaining only edges whose event
    # count is >= threshold.
    eids, nodes_list, times_list = [], [], []
    with open(timing_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != k:
                continue
            eid = int(parts[0])
            if eid not in nodes_of:
                continue
            ts = parts[2:]
            if len(ts) < threshold:
                continue
            ts_int = np.fromiter((int(x) for x in ts), dtype=np.int64)
            eids.append(eid)
            nodes_list.append(nodes_of[eid])
            times_list.append(ts_int)
    return eids, nodes_list, times_list


def load_size2(folder, threshold):
    """Specialisation of `load_size_k` for k=2 dyadic edges.

    Returns the same three lists as `load_size_k`, but with the second
    list flattened to plain `(u, v)` tuples (no sorting beyond what
    convert_*.py already did) — slightly faster for the size-2 path
    which is the most-used.
    """
    list_path = os.path.join(folder, "edges_list.txt")
    timing_path = os.path.join(folder, "edges_timings.txt")

    nodes_of = {}
    with open(list_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != 2:
                continue
            eid = int(parts[0])
            nodes_of[eid] = (int(parts[2]), int(parts[3]))

    eids, nodes_list, times_list = [], [], []
    with open(timing_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != 2:
                continue
            eid = int(parts[0])
            if eid not in nodes_of:
                continue
            ts = parts[2:]
            if len(ts) < threshold:
                continue
            ts_int = np.fromiter((int(x) for x in ts), dtype=np.int64)
            eids.append(eid)
            nodes_list.append(nodes_of[eid])
            times_list.append(ts_int)
    return eids, nodes_list, times_list


def load_static_size2_edges(folder):
    """Return the set of *all* dyadic edges in the static aggregated
    network, regardless of event count.

    Used to test whether two unshared endpoints v1, v2 of an
    intersecting size-2 pair form a recorded edge — i.e., whether the
    pair is intersecting-closed or intersecting-open. Note we ignore
    hyperedges of size > 2 here, per the v32 clarification (see
    section 3.1.2 of the paper).
    """
    list_path = os.path.join(folder, "edges_list.txt")
    edges = set()
    with open(list_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != 2:
                continue
            u, v = int(parts[2]), int(parts[3])
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return edges


# ---------------------------------------------------------------------------
# 2. Time-binning — Eq. (1) in the paper
# ---------------------------------------------------------------------------

def aggregate(times_list, T, delta):
    """Bin event timestamps into non-overlapping windows of width Δ.

    Implements \\bar e_i(\\ell) = #{t in [(\\ell-1)Δ, \\ell Δ) : event i
    occurs at t}, returning an N x ceil(T/Δ) integer matrix of these
    counts. Multiple events at the same timestamp accumulate.

    Parameters
    ----------
    times_list : list of np.ndarray
        Per-edge timestamps as returned by `load_size_k` / `load_size2`.
    T : int
        Total observation length in native time units (i.e., the
        smallest integer such that all events satisfy t < T).
    delta : int
        Time-window width Δ in native time units.
    """
    n_bins = int(np.ceil(T / delta))
    N = len(times_list)
    X = np.zeros((N, n_bins), dtype=np.int32)
    for i, ts in enumerate(times_list):
        b = (ts // delta).astype(np.int64)
        np.add.at(X[i], b, 1)
    return X


# ---------------------------------------------------------------------------
# 3. EEC math — Eq. (2) [cosine, main text] and Eq. (5) [Pearson, appendix]
# ---------------------------------------------------------------------------

def corr_matrix_pearson(X):
    """Return the N x N Pearson-correlation EEC matrix (Eq. (5)).

    Mean-centres each row of X before taking the inner product. The
    resulting EEC takes values in [-1, 1].
    """
    Xf = X.astype(np.float64)
    Xc = Xf - Xf.mean(axis=1, keepdims=True)
    norms = np.sqrt((Xc * Xc).sum(axis=1))
    norms[norms == 0] = 1.0
    Xn = Xc / norms[:, None]
    C = Xn @ Xn.T
    np.clip(C, -1.0, 1.0, out=C)
    return C


def corr_matrix_cosine(X):
    """Return the N x N cosine-similarity EEC matrix (Eq. (2)).

    Identical to `corr_matrix_pearson` except no mean-centring; the
    resulting EEC takes values in [0, 1] because event counts are
    non-negative.
    """
    Xf = X.astype(np.float64)
    norms = np.sqrt((Xf * Xf).sum(axis=1))
    norms[norms == 0] = 1.0
    Xn = Xf / norms[:, None]
    C = Xn @ Xn.T
    np.clip(C, -1.0, 1.0, out=C)
    return C


def cross_corr_pearson(Xa, Xb):
    """Pearson EEC for cross pairs of (size-2, size-3) hyperedges.

    Returns a Na x Nb matrix where entry (i, j) is the Pearson EEC
    between edge i (from Xa) and hyperedge j (from Xb). Used by
    section 4 of the paper for the e-h3 analysis.
    """
    Xa = Xa.astype(np.float64)
    Xb = Xb.astype(np.float64)
    Xac = Xa - Xa.mean(axis=1, keepdims=True)
    Xbc = Xb - Xb.mean(axis=1, keepdims=True)
    na = np.sqrt((Xac * Xac).sum(axis=1))
    nb = np.sqrt((Xbc * Xbc).sum(axis=1))
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    Xan = Xac / na[:, None]
    Xbn = Xbc / nb[:, None]
    C = Xan @ Xbn.T
    np.clip(C, -1.0, 1.0, out=C)
    return C


def cross_corr_cosine(Xa, Xb):
    """Cosine-similarity EEC for cross pairs of (size-2, size-3) hyperedges."""
    Xa = Xa.astype(np.float64)
    Xb = Xb.astype(np.float64)
    na = np.sqrt((Xa * Xa).sum(axis=1))
    nb = np.sqrt((Xb * Xb).sum(axis=1))
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    Xan = Xa / na[:, None]
    Xbn = Xb / nb[:, None]
    C = Xan @ Xbn.T
    np.clip(C, -1.0, 1.0, out=C)
    return C


# ---------------------------------------------------------------------------
# 4. Pair classification — # shared nodes and triangle-closure test
# ---------------------------------------------------------------------------

def shared_node_matrix(node_tuples):
    """Return an N x N int8 matrix of shared-node counts.

    Entry (i, j) is the number of nodes that hyperedges i and j have in
    common. Used to classify pairs as disjoint (0), 1-intersecting (1),
    or 2-intersecting (2).
    """
    N = len(node_tuples)
    all_nodes = sorted(set().union(*node_tuples))
    idx = {n: k for k, n in enumerate(all_nodes)}
    M = np.zeros((N, len(all_nodes)), dtype=np.int8)
    for i, ns in enumerate(node_tuples):
        for n in ns:
            M[i, idx[n]] = 1
    # int16 multiplication, then cast back; the largest possible shared
    # count for size-3 hyperedges is 3.
    return (M.astype(np.int16) @ M.T.astype(np.int16)).astype(np.int8)


def build_cooccurrence(folder, V_kept):
    """Return a dict mapping each kept node to the set of nodes that
    co-occur with it in some static hyperedge of any size.

    This is the lookup used by the triangle-closure test for the
    intersecting-open vs intersecting-closed split. Specifically, an
    intersecting size-2 pair {(v, v1), (v, v2)} is intersecting-closed
    iff (v1, v2) is a recorded dyadic edge in the data; analogous tests
    apply to e-h3 and h3-h3 pairs.

    The implementation builds a sparse incidence matrix M of shape
    (|V_kept|, num_static_hyperedges) and computes M M^T to obtain the
    co-occurrence relation in one step.

    Parameters
    ----------
    folder : str
        Dataset folder, same convention as `load_size_k`.
    V_kept : list of int
        The nodes that appear in at least one *frequent* edge — only
        these are queried during pair classification, so we restrict
        the co-occurrence relation to them to save memory.
    """
    Vk = set(V_kept)
    v_to_idx = {v: i for i, v in enumerate(V_kept)}
    rows, cols = [], []
    eid_counter = 0
    list_path = os.path.join(folder, "edges_list.txt")
    with open(list_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            ns = parts[2:2 + sz]
            in_kept = [v_to_idx[int(x)] for x in ns if int(x) in Vk]
            if len(in_kept) < 2:
                eid_counter += 1
                continue
            for vi in in_kept:
                rows.append(vi)
                cols.append(eid_counter)
            eid_counter += 1
    if not rows:
        return {v: set() for v in V_kept}
    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(V_kept), eid_counter),
    )
    C = (M @ M.T).tocoo()
    coo_dict = {v: set() for v in V_kept}
    for u, v in zip(C.row, C.col):
        if u == v:
            continue
        coo_dict[V_kept[u]].add(V_kept[v])
    return coo_dict
