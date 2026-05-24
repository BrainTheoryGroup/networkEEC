"""Main analysis pipeline for the EEC paper (`networkEEC_35.tex`).

For each (dataset, pair-type) combination this script:

  1. loads the time-stamped event sequences using `eec_helpers.load_*`;
  2. bins them into windows of width Δ (per-dataset value, see DELTA);
  3. computes the EEC matrix between every frequent edge / hyperedge;
  4. classifies each pair by # shared nodes and (for 1- and 2-shared
     pairs) by triangle closure;
  5. writes per-pair EEC arrays to a `.npz` file when the dataset has
     enough samples to support a comparison;
  6. writes one row of summary statistics (means, p-values, Cohen's d)
     to `summary.tsv`.

The same pipeline is used twice — once with similarity = "pearson",
once with similarity = "cosine" — and writes its outputs into
`out/v9/` and `out/v9_eec2/` respectively. The results in `out/v9/`
populate the appendix tables (Pearson EEC); the results in
`out/v9_eec2/` populate the main-text tables and figures (cosine EEC).

Run from the project root:

    python3 code/analysis/compile_v9.py             # Pearson (default)
    python3 code/analysis/compile_v9.py cosine      # cosine similarity

The output `.npz` files are then consumed by:
  * `code/figures_tables/render_figs_v9.py` (Fig 2, 3, 5),
  * `code/figures_tables/render_tables_v9.py` (Tables 2, 3, 4 in the
    main text, and the values appearing in Tables 5, 6, 7 in the
    appendix; the appendix tables themselves are hand-written from
    the EMD analysis in `compute_emd_distances.py` and from the
    Pearson summary.tsv produced by this script).
"""

import os
import sys
import time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eec_helpers import (
    load_size2, load_size_k, load_static_size2_edges,
    aggregate, build_cooccurrence, shared_node_matrix,
    corr_matrix_pearson, corr_matrix_cosine,
    cross_corr_pearson, cross_corr_cosine,
)

# ---------------------------------------------------------------------------
# Paths and per-dataset configuration
# ---------------------------------------------------------------------------

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Pairwise (genuinely temporal-network) datasets and the temporal hypergraphs
# live in different subfolders of `data/`. The set of pairwise slugs is
# fixed; everything else listed in DELTA below is a temporal hypergraph.
PAIRWISE_NETWORKS = {
    "ia-reality-call",
    "copenhagen-calls",
    "college-msg",
    "friends-family-call",
    "social-evolution-call",
}


def dataset_folder(name):
    """Return the absolute path to a dataset folder for the given slug."""
    sub = "networks" if name in PAIRWISE_NETWORKS else "hypergraphs"
    return os.path.join(ROOT, "data", sub, name)


# Frequent-edge thresholds θ. Section 3.1.1: at least θ_2 = 50 events on
# size-2 (dyadic) edges; section 4.1.1: at least θ_3 = 25 events on
# size-3 hyperedges (lower because size-3 events are inherently rarer).
THETA_2 = 50
THETA_3 = 25

# Per-hypergraph minimum on the count of frequent hyperedges entering the
# e-h3 and h3-h3 analyses: E_{2,freq}+E_{3,freq} >= 40 for e-h3 and
# E_{3,freq} >= 40 for h3-h3.
MIN_HYPER_FREQ = 40

# Per-dataset time-window width Δ in the dataset's native time unit.
# Chosen to give roughly T/Δ ≈ 200 bins, except for already-coarse
# datasets where Δ = 1 (e.g. DAWN at quarter-year resolution). The
# values shown in column "Δ" of Table 2 of the paper.
DELTA = {
    # Five pairwise temporal networks.
    "ia-reality-call":         3600,    # 1 hour
    "copenhagen-calls":        3600,
    "college-msg":             86400,   # 1 day
    "friends-family-call":     86400,
    "social-evolution-call":   86400,
    # Seventeen temporal hypergraphs.
    "contact-high-school":     90,
    "contact-primary-school":  30,
    "DAWN":                    1,
    "NDC-classes":             200,
    "NDC-substances":          200,
    "coauth-DBLP":             1,
    "coauth-MAG-Geology":      1,
    "coauth-MAG-History":      1,
    "congress-bills":          60,
    "email-Enron":             7,
    "email-Eu":                86400,
    "tags-ask-ubuntu":         400,
    "tags-math-sx":            325,
    "tags-stack-overflow":     400,
    "threads-math-sx":         311,
    "threads-stack-overflow":  400,
}
DATASETS = list(DELTA.keys())


# Output directories per similarity choice. Both folders carry an
# identical schema: summary.tsv, summary_3x3.tsv, plus per-dataset
# .npz arrays. Downstream renderers read from these.
OUT_DIRS = {
    "pearson": os.path.join(ROOT, "out", "v9"),
    "cosine":  os.path.join(ROOT, "out", "v9_eec2"),
}


# ---------------------------------------------------------------------------
# Cohen's d and per-comparison statistics
# ---------------------------------------------------------------------------

def cohen_d(a, b):
    """Cohen's d using the unbiased (sample-sd) pooled standard deviation.

    Returns (mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2). NaN if
    either sample has fewer than two observations or the pooled sd is
    zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def stat_pack(name, pair_type, delta, r_dj, r_open, r_tri, n_sh2=None):
    """Build a row dict of summary statistics for the size-2 / size-2x3 case.

    `r_dj` are EEC values for disjoint pairs;
    `r_open` and `r_tri` are intersecting pairs split by the triangle
    closure test, called intersecting-open and intersecting-closed in
    the paper. Reported quantities:
      - sample sizes per class;
      - per-class means;
      - two-sided Welch t-test p-value (and Mann-Whitney U for the
        two-class comparison) and Cohen's d for each comparison.
    """
    n_dj = len(r_dj); n_open = len(r_open); n_tri = len(r_tri)
    r_ns = (np.concatenate([r_open, r_tri])
            if (n_open or n_tri) else np.array([], dtype=np.float64))
    n_ns = len(r_ns)
    out = {
        "dataset": name, "pair_type": pair_type, "delta": delta,
        "n_dj": n_dj, "n_open": n_open, "n_tri": n_tri, "n_ns": n_ns,
        "n_sh2": n_sh2 if n_sh2 is not None else 0,
        "mean_dj":   float(r_dj.mean())   if n_dj   else float("nan"),
        "mean_ns":   float(r_ns.mean())   if n_ns   else float("nan"),
        "mean_open": float(r_open.mean()) if n_open else float("nan"),
        "mean_tri":  float(r_tri.mean())  if n_tri  else float("nan"),
        "p_dj_ns_welch": float("nan"),
        "p_dj_ns_mwu":   float("nan"),
        "d_dj_ns":       float("nan"),
        "p_dj_open":  float("nan"), "p_dj_tri":  float("nan"),
        "p_open_tri": float("nan"),
        "d_dj_open":  float("nan"), "d_dj_tri":  float("nan"),
        "d_open_tri": float("nan"),
    }
    if n_dj >= 2 and n_ns >= 2:
        out["p_dj_ns_welch"] = float(
            stats.ttest_ind(r_ns, r_dj, equal_var=False).pvalue)
        out["p_dj_ns_mwu"] = float(
            stats.mannwhitneyu(r_ns, r_dj, alternative="two-sided").pvalue)
        out["d_dj_ns"] = cohen_d(r_ns, r_dj)
    if n_dj >= 2 and n_open >= 2:
        out["p_dj_open"] = float(
            stats.ttest_ind(r_open, r_dj, equal_var=False).pvalue)
        out["d_dj_open"] = cohen_d(r_open, r_dj)
    if n_dj >= 2 and n_tri >= 2:
        out["p_dj_tri"] = float(
            stats.ttest_ind(r_tri, r_dj, equal_var=False).pvalue)
        out["d_dj_tri"] = cohen_d(r_tri, r_dj)
    if n_open >= 2 and n_tri >= 2:
        out["p_open_tri"] = float(
            stats.ttest_ind(r_tri, r_open, equal_var=False).pvalue)
        out["d_open_tri"] = cohen_d(r_tri, r_open)
    return out


def stat_pack_3x3(name, delta, r_dj,
                  r_ns1_open, r_ns1_tri, r_ns2_open, r_ns2_tri):
    """Per-row stats for the size-3x3 case.

    Two intersecting subclasses (1-intersecting and 2-intersecting)
    times two triangle-closure subclasses gives four "fine" classes
    plus the disjoint baseline; we report means and Cohen's d for
    every comparison cited in the paper.
    """
    r_ns1 = (np.concatenate([r_ns1_open, r_ns1_tri])
             if (len(r_ns1_open) or len(r_ns1_tri))
             else np.array([], dtype=np.float64))
    r_ns2 = (np.concatenate([r_ns2_open, r_ns2_tri])
             if (len(r_ns2_open) or len(r_ns2_tri))
             else np.array([], dtype=np.float64))

    def m(a):
        return float(a.mean()) if len(a) else float("nan")

    def pd(a, b):
        if len(a) >= 2 and len(b) >= 2:
            return (
                float(stats.ttest_ind(a, b, equal_var=False).pvalue),
                cohen_d(a, b),
            )
        return float("nan"), float("nan")

    out = {
        "dataset": name, "pair_type": "3x3", "delta": delta,
        "n_dj": len(r_dj),
        "n_ns1": len(r_ns1), "n_ns2": len(r_ns2),
        "n_ns1_open": len(r_ns1_open), "n_ns1_tri": len(r_ns1_tri),
        "n_ns2_open": len(r_ns2_open), "n_ns2_tri": len(r_ns2_tri),
        "mean_dj":       m(r_dj),
        "mean_ns1":      m(r_ns1),       "mean_ns2":      m(r_ns2),
        "mean_ns1_open": m(r_ns1_open),  "mean_ns1_tri":  m(r_ns1_tri),
        "mean_ns2_open": m(r_ns2_open),  "mean_ns2_tri":  m(r_ns2_tri),
    }
    out["p_dj_ns1"],       out["d_ns1_dj"]      = pd(r_ns1, r_dj)
    out["p_dj_ns2"],       out["d_ns2_dj"]      = pd(r_ns2, r_dj)
    out["p_ns1_ns2"],      out["d_ns2_ns1"]     = pd(r_ns2, r_ns1)
    out["p_dj_ns1_open"],  out["d_ns1_open_dj"] = pd(r_ns1_open, r_dj)
    out["p_dj_ns1_tri"],   out["d_ns1_tri_dj"]  = pd(r_ns1_tri,  r_dj)
    out["p_dj_ns2_open"],  out["d_ns2_open_dj"] = pd(r_ns2_open, r_dj)
    out["p_dj_ns2_tri"],   out["d_ns2_tri_dj"]  = pd(r_ns2_tri,  r_dj)
    return out


# ---------------------------------------------------------------------------
# Pair categorisation — three flavours (size-2, size-2x3, size-3x3)
# ---------------------------------------------------------------------------

def categorize_2x2(folder, name, delta, similarity):
    """Size-2 vs size-2 pairs.

    Procedure:
      1. Load all size-2 edges with at least θ_2 = 50 events.
      2. Aggregate event timestamps into windows of width Δ.
      3. Build the EEC matrix C using the chosen similarity.
      4. For each upper-triangular pair (i, j) classify by # shared
         nodes; for intersecting pairs (1 shared) test whether the
         unshared endpoints (v1, v2) form a recorded dyadic edge in
         the static aggregated network — if so, the pair is
         intersecting-closed (`r_tri`), otherwise intersecting-open
         (`r_open`).

    Returns three 1-D float arrays of EEC values: r_dj, r_open, r_tri.
    """
    eids, nodes_list, times_list = load_size2(folder, THETA_2)
    N = len(eids)
    if N < 2:
        return None
    T = int(max(int(ts.max()) for ts in times_list)) + 1
    X = aggregate(times_list, T, delta)
    C = corr_matrix_pearson(X) if similarity == "pearson" \
        else corr_matrix_cosine(X)

    a = np.array([n[0] for n in nodes_list], dtype=np.int64)
    b = np.array([n[1] for n in nodes_list], dtype=np.int64)
    static_edges = load_static_size2_edges(folder)

    iu = np.triu_indices(N, k=1)
    r_arr = C[iu]

    S = shared_node_matrix(nodes_list)
    s_arr = S[iu]

    iu_i, iu_j = iu
    r_dj = r_arr[s_arr == 0]            # disjoint pairs
    ns_idx = np.where(s_arr == 1)[0]    # intersecting (1 shared node)
    r_open, r_tri = [], []
    for k in ns_idx:
        i, j = int(iu_i[k]), int(iu_j[k])
        ai, bi = int(a[i]), int(b[i])
        aj, bj = int(a[j]), int(b[j])
        # Identify the shared node (the only node in both edges) and
        # the two unshared endpoints v1, v2.
        shared = ai if (ai == aj or ai == bj) else bi
        u1 = bi if shared == ai else ai
        u2 = bj if shared == aj else aj
        u, w = (u1, u2) if u1 < u2 else (u2, u1)
        if (u, w) in static_edges:
            r_tri.append(r_arr[k])
        else:
            r_open.append(r_arr[k])
    return (np.array(r_dj, dtype=np.float64),
            np.array(r_open, dtype=np.float64),
            np.array(r_tri, dtype=np.float64))


def categorize_2x3(folder, name, delta, similarity):
    """e-h3 (cross-size) pairs: a size-2 edge versus a size-3 hyperedge.

    Disjoint = no shared node; intersecting = exactly one shared node.
    Pairs sharing two nodes are excluded (the size-2 edge is then a
    sub-edge of the size-3 hyperedge). Triangle closure is tested via
    the static co-occurrence relation.
    """
    e2_eids, n2, t2 = load_size_k(folder, 2, THETA_2)
    e3_eids, n3, t3 = load_size_k(folder, 3, THETA_3)
    Na, Nb = len(e2_eids), len(e3_eids)
    if Na < 1 or Nb < 1 or Na + Nb < MIN_HYPER_FREQ:
        return None
    T = max(
        int(max(int(ts.max()) for ts in t2)) if t2 else 0,
        int(max(int(ts.max()) for ts in t3)) if t3 else 0,
    ) + 1
    Xa = aggregate(t2, T, delta)
    Xb = aggregate(t3, T, delta)
    C = (cross_corr_pearson(Xa, Xb) if similarity == "pearson"
         else cross_corr_cosine(Xa, Xb))

    V_kept = sorted(
        set(v for tup in n2 for v in tup) |
        set(v for tup in n3 for v in tup)
    )
    coo = build_cooccurrence(folder, V_kept)
    r_dj, r_open, r_tri = [], [], []
    for i in range(Na):
        e1 = n2[i]; s1 = set(e1)
        for j in range(Nb):
            e2 = n3[j]; s2 = set(e2)
            shared = s1 & s2
            if not shared:
                r_dj.append(C[i, j])
                continue
            if len(shared) >= 2:
                # size-2 edge sits inside size-3 hyperedge, exclude
                continue
            x = next(iter(shared))
            v1 = e1[0] if e1[0] != x else e1[1]
            v2, v3 = [v for v in e2 if v != x]
            v1_co = coo.get(v1, ())
            # intersecting-closed iff the unshared endpoint of e1 co-occurs
            # with at least one unshared endpoint of e2 in some static
            # hyperedge of any size.
            if v2 in v1_co or v3 in v1_co:
                r_tri.append(C[i, j])
            else:
                r_open.append(C[i, j])
    return (np.array(r_dj, dtype=np.float64),
            np.array(r_open, dtype=np.float64),
            np.array(r_tri, dtype=np.float64))


def categorize_3x3(folder, name, delta, similarity):
    """h3-h3 pairs of size-3 hyperedges.

    Two size-3 hyperedges may share 0, 1, or 2 nodes (3 means the same
    hyperedge, excluded). Returns five arrays of EEC values:
      r_dj         : 0 shared nodes
      r_ns1_open   : 1 shared node, no triangle closure on unshared pair
      r_ns1_tri    : 1 shared node, triangle closed
      r_ns2_open   : 2 shared nodes, no triangle closure on unshared pair
      r_ns2_tri    : 2 shared nodes, triangle closed
    """
    eids, nodes_list, times_list = load_size_k(folder, 3, THETA_3)
    N = len(eids)
    if N < MIN_HYPER_FREQ:
        return None
    T = int(max(int(ts.max()) for ts in times_list)) + 1
    X = aggregate(times_list, T, delta)
    C = corr_matrix_pearson(X) if similarity == "pearson" \
        else corr_matrix_cosine(X)
    S = shared_node_matrix(nodes_list)
    iu = np.triu_indices(N, k=1)
    s_arr = S[iu]; r_arr = C[iu]
    iu_i, iu_j = iu
    V_kept = sorted(set(v for tup in nodes_list for v in tup))
    coo = build_cooccurrence(folder, V_kept)
    r_dj = []
    r_ns1_open, r_ns1_tri = [], []
    r_ns2_open, r_ns2_tri = [], []
    for k in range(s_arr.shape[0]):
        sh = int(s_arr[k])
        if sh == 0:
            r_dj.append(r_arr[k])
            continue
        if sh >= 3:
            continue  # same hyperedge, exclude
        i, j = int(iu_i[k]), int(iu_j[k])
        e1 = nodes_list[i]; e2 = nodes_list[j]
        s1 = set(e1); s2 = set(e2)
        u1 = [v for v in e1 if v not in s2]   # unshared nodes of e1
        u2 = [v for v in e2 if v not in s1]   # unshared nodes of e2
        any_co = False
        for v in u1:
            v_co = coo.get(v, ())
            if not v_co:
                continue
            for w in u2:
                if w in v_co:
                    any_co = True
                    break
            if any_co:
                break
        if sh == 1:
            (r_ns1_tri if any_co else r_ns1_open).append(r_arr[k])
        else:  # sh == 2
            (r_ns2_tri if any_co else r_ns2_open).append(r_arr[k])
    return tuple(np.array(x, dtype=np.float64) for x in
                 (r_dj, r_ns1_open, r_ns1_tri, r_ns2_open, r_ns2_tri))


# ---------------------------------------------------------------------------
# Disk I/O — `.npz` arrays and `summary.tsv`
# ---------------------------------------------------------------------------

# Power-filter thresholds (also reproduced in the figure / table renderers).
DJ_NS_THRESH    = 20   # n_dj >= 20 AND n_ns >= 20  for the dj-vs-int test
THREE_OPEN_TRI  = 10   # n_open, n_tri >= 10 each   for the three-class test


def maybe_save_arrays(out_v9, name, pair_type, r_dj, r_open, r_tri):
    """Save (r_dj, r_open, r_tri) iff at least one downstream comparison
    has enough samples to be meaningful (see DJ_NS_THRESH / THREE_OPEN_TRI)."""
    n_dj = len(r_dj); n_open = len(r_open); n_tri = len(r_tri)
    n_ns = n_open + n_tri
    save_djns  = (n_dj >= DJ_NS_THRESH and n_ns >= DJ_NS_THRESH)
    save_three = (n_dj >= DJ_NS_THRESH and n_open >= THREE_OPEN_TRI
                  and n_tri >= THREE_OPEN_TRI)
    if save_djns or save_three:
        np.savez_compressed(
            os.path.join(out_v9, f"{name}_{pair_type}.npz"),
            r_dj=r_dj.astype(np.float32),
            r_open=r_open.astype(np.float32),
            r_tri=r_tri.astype(np.float32),
        )


def maybe_save_arrays_3x3(out_v9, name, r_dj,
                          r_ns1_open, r_ns1_tri, r_ns2_open, r_ns2_tri):
    """Save five arrays for the size-3x3 case iff the main or fine power
    filter is satisfied (see compile docstring)."""
    n_dj = len(r_dj)
    n_ns1 = len(r_ns1_open) + len(r_ns1_tri)
    n_ns2 = len(r_ns2_open) + len(r_ns2_tri)
    save_main = (n_dj >= DJ_NS_THRESH and n_ns1 >= DJ_NS_THRESH
                 and n_ns2 >= DJ_NS_THRESH)
    save_fine = (n_dj >= DJ_NS_THRESH
                 and len(r_ns1_open) >= THREE_OPEN_TRI
                 and len(r_ns1_tri) >= THREE_OPEN_TRI
                 and len(r_ns2_open) >= THREE_OPEN_TRI
                 and len(r_ns2_tri) >= THREE_OPEN_TRI)
    if save_main or save_fine:
        np.savez_compressed(
            os.path.join(out_v9, f"{name}_3x3.npz"),
            r_dj=r_dj.astype(np.float32),
            r_ns1_open=r_ns1_open.astype(np.float32),
            r_ns1_tri=r_ns1_tri.astype(np.float32),
            r_ns2_open=r_ns2_open.astype(np.float32),
            r_ns2_tri=r_ns2_tri.astype(np.float32),
        )


SUMMARY_2_FIELDS = [
    "dataset", "pair_type", "delta",
    "n_dj", "n_open", "n_tri", "n_ns", "n_sh2",
    "mean_dj", "mean_ns", "mean_open", "mean_tri",
    "p_dj_ns_welch", "p_dj_ns_mwu", "d_dj_ns",
    "p_dj_open", "p_dj_tri", "p_open_tri",
    "d_dj_open", "d_dj_tri", "d_open_tri",
]
SUMMARY_3X3_FIELDS = [
    "dataset", "pair_type", "delta",
    "n_dj", "n_ns1", "n_ns2",
    "n_ns1_open", "n_ns1_tri", "n_ns2_open", "n_ns2_tri",
    "mean_dj", "mean_ns1", "mean_ns2",
    "mean_ns1_open", "mean_ns1_tri", "mean_ns2_open", "mean_ns2_tri",
    "p_dj_ns1", "p_dj_ns2", "p_ns1_ns2",
    "d_ns1_dj", "d_ns2_dj", "d_ns2_ns1",
    "p_dj_ns1_open", "p_dj_ns1_tri", "p_dj_ns2_open", "p_dj_ns2_tri",
    "d_ns1_open_dj", "d_ns1_tri_dj", "d_ns2_open_dj", "d_ns2_tri_dj",
]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def main(similarity="pearson"):
    """Run the full pipeline for every dataset in DELTA, writing outputs
    into the OUT_DIRS[similarity] folder."""
    if similarity not in OUT_DIRS:
        raise ValueError(f"unknown similarity: {similarity!r}")
    out_v9 = OUT_DIRS[similarity]
    os.makedirs(out_v9, exist_ok=True)
    print(f"[similarity={similarity}] writing outputs to {out_v9}",
          flush=True)

    sum_path = os.path.join(out_v9, "summary.tsv")
    sum_3x3_path = os.path.join(out_v9, "summary_3x3.tsv")

    with open(sum_3x3_path, "w") as fh3, open(sum_path, "w") as fh:
        fh3.write("\t".join(SUMMARY_3X3_FIELDS) + "\n")
        fh.write("\t".join(SUMMARY_2_FIELDS) + "\n")
        for name in DATASETS:
            folder = dataset_folder(name)
            delta = DELTA[name]
            print(f"\n=== {name}  Δ={delta} ===", flush=True)

            # 2x2 (size-2 vs size-2)
            t0 = time.time()
            try:
                out = categorize_2x2(folder, name, delta, similarity)
            except Exception as e:
                print(f"  2x2 error: {e}"); out = None
            if out is not None:
                r_dj, r_open, r_tri = out
                stats_d = stat_pack(name, "2x2", delta, r_dj, r_open, r_tri)
                fh.write("\t".join(str(stats_d[k])
                                   for k in SUMMARY_2_FIELDS) + "\n")
                fh.flush()
                maybe_save_arrays(out_v9, name, "2x2", r_dj, r_open, r_tri)
                print(f"  2x2: n_dj={len(r_dj)} n_open={len(r_open)} "
                      f"n_tri={len(r_tri)}  ({time.time()-t0:.1f}s)")

            # 2x3 (size-2 vs size-3 cross)
            t0 = time.time()
            try:
                out = categorize_2x3(folder, name, delta, similarity)
            except Exception as e:
                print(f"  2x3 error: {e}"); out = None
            if out is not None:
                r_dj, r_open, r_tri = out
                stats_d = stat_pack(name, "2x3", delta, r_dj, r_open, r_tri)
                fh.write("\t".join(str(stats_d[k])
                                   for k in SUMMARY_2_FIELDS) + "\n")
                fh.flush()
                maybe_save_arrays(out_v9, name, "2x3", r_dj, r_open, r_tri)
                print(f"  2x3: n_dj={len(r_dj)} n_open={len(r_open)} "
                      f"n_tri={len(r_tri)}  ({time.time()-t0:.1f}s)")

            # 3x3 (size-3 vs size-3)
            t0 = time.time()
            try:
                out = categorize_3x3(folder, name, delta, similarity)
            except Exception as e:
                print(f"  3x3 error: {e}"); out = None
            if out is not None:
                r_dj, r_ns1_open, r_ns1_tri, r_ns2_open, r_ns2_tri = out
                sd = stat_pack_3x3(name, delta, r_dj, r_ns1_open, r_ns1_tri,
                                   r_ns2_open, r_ns2_tri)
                fh3.write("\t".join(str(sd.get(k, "nan"))
                                    for k in SUMMARY_3X3_FIELDS) + "\n")
                fh3.flush()
                maybe_save_arrays_3x3(out_v9, name, r_dj, r_ns1_open,
                                      r_ns1_tri, r_ns2_open, r_ns2_tri)
                n_ns1 = len(r_ns1_open) + len(r_ns1_tri)
                n_ns2 = len(r_ns2_open) + len(r_ns2_tri)
                print(f"  3x3: n_dj={len(r_dj)} n_ns1={n_ns1} "
                      f"n_ns2={n_ns2} (open/tri ns1={len(r_ns1_open)}/"
                      f"{len(r_ns1_tri)} ns2={len(r_ns2_open)}/"
                      f"{len(r_ns2_tri)})  ({time.time()-t0:.1f}s)")

    print(f"\nWrote {sum_path}")
    print(f"Wrote {sum_3x3_path}")


if __name__ == "__main__":
    sim = sys.argv[1] if len(sys.argv) > 1 else "pearson"
    main(sim)
