"""Sihn--Kim (2019) Earth Mover's Distance (EMD) analysis.

Computes the EMD-based concurrency measure for the same per-pair-type
comparisons as Tables 2, 3 and 4 of `networkEEC_35.tex`. Reports
Cohen's d only — the per-class means are scale-dependent on the
EMD-of-timestamps and not directly comparable to those of EEC.

Method (Sihn & Kim 2019):
  Given two spike trains f, g, view each as a discrete probability
  distribution that places mass 1/N at every spike timestamp, where N
  is the train's spike count. The Sihn--Kim distance is the
  1D Wasserstein-1 distance between these two distributions with
  ground metric d(x, y) = |x - y|. The method is parameter-free.

Sign convention:
  EMD is a *distance*: smaller value → more concurrent. Therefore
  d_int_dj < 0 means intersecting pairs are more concurrent than
  disjoint pairs (opposite sign to the cosine / Pearson EEC).

Outputs:
  Three LaTeX tabulars (d-only, no μ values) into out/v9_eec2/:
    TABLE_emd_repro_T2.tex   size-2: d, d_open-dj, d_closed-dj
    TABLE_emd_repro_T3.tex   e-h3:   d, d_open-dj, d_closed-dj
    TABLE_emd_repro_T4.tex   h3-h3:  d_int1-dj, d_int2-int1
  These are the source values for the EMD half of Tables 5, 6, and 7
  in the appendix of `networkEEC_35.tex`. The actual appendix tables
  in `out/TABLE_app_size{2,2x3,3x3}_pcc_emd.tex` are hand-written —
  they combine these EMD d's with Pearson d's read from
  `out/v9/summary.tsv`.

To keep wall-clock manageable on the largest datasets we randomly
subsample at most MAX_PAIRS_PER_CLASS pairs per class per dataset
(seed 20260502). Smaller classes are used in full.

Run from the project root:
    python3 code/analysis/sihnkim_emd_full.py
"""

import os
import sys
import time
import numpy as np
from scipy.stats import wasserstein_distance

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from eec_helpers import load_size2, load_size_k, build_cooccurrence
import compile_v9 as cv9


ROOT = os.path.join(CODE_DIR, "..", "..")

# Same θ thresholds as in the EEC pipeline.
THETA2 = 50
THETA3 = 25
MAX_PAIRS_PER_CLASS = 5000
RNG = np.random.default_rng(20260502)


# Datasets evaluated for each table. These mirror the rows that
# survive the manuscript's power filter (see Tables 2/3/4 of v35).
DATASETS_T2 = [
    "ia-reality-call",
    "college-msg",
    "friends-family-call",
    "contact-high-school", "contact-primary-school",
    "DAWN", "congress-bills", "email-Eu", "tags-ask-ubuntu",
    "tags-math-sx", "tags-stack-overflow",
]
DATASETS_T3 = [
    "contact-high-school", "contact-primary-school", "DAWN",
    "NDC-substances", "email-Eu", "tags-ask-ubuntu", "tags-math-sx",
    "tags-stack-overflow",
]
DATASETS_T4 = [
    "DAWN", "NDC-substances", "email-Eu",
    "tags-ask-ubuntu", "tags-math-sx", "tags-stack-overflow",
]


# ---------------------------------------------------------------------------
# Per-pair distance and per-comparison Cohen's d
# ---------------------------------------------------------------------------

def emd(tx, ty):
    """1D Wasserstein-1 between two timestamp arrays with uniform
    mass 1/N per spike, as in Sihn--Kim 2019."""
    return float(wasserstein_distance(tx, ty))


def cohens_d(x, y):
    """(mean(x) - mean(y)) / sqrt((var(x) + var(y)) / 2). NaN if a
    sample has fewer than 2 points or the pooled sd is zero."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    p = np.sqrt(0.5 * (sx ** 2 + sy ** 2))
    if p == 0:
        return float("nan")
    return (float(np.mean(x)) - float(np.mean(y))) / p


def fmt_d(x):
    """Cohen's d formatter for LaTeX cells (negatives wrapped in $...$)."""
    if x != x:
        return "--"
    return f"${x:.2f}$" if x < 0 else f"{x:.2f}"


def take(lst, k):
    """Random subsample of up to k entries from `lst`, deterministic
    given the module-level RNG seed."""
    if len(lst) <= k:
        return list(lst)
    idx = RNG.choice(len(lst), size=k, replace=False)
    return [lst[i] for i in idx]


def emd_for_pairs(times_a, times_b_list, pairs):
    """Compute the EMD for a list of (i, j) pairs."""
    out = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        out[k] = emd(times_a[i], times_b_list[j])
    return out


# ---------------------------------------------------------------------------
# Table-2 (size-2 vs size-2 pairs)
# ---------------------------------------------------------------------------

def run_t2(name):
    """Compute d, d_open-dj, d_closed-dj for size-2 pairs of one dataset."""
    folder = cv9.dataset_folder(name)
    eids, nodes_list, times_list = load_size2(folder, THETA2)
    n = len(eids)
    a = np.array([nl[0] for nl in nodes_list], dtype=np.int64)
    b = np.array([nl[1] for nl in nodes_list], dtype=np.int64)
    V_kept = sorted(set(a.tolist()) | set(b.tolist()))
    coo = build_cooccurrence(folder, V_kept)

    # Enumerate all unordered pairs of frequent edges and classify them
    # by # shared nodes. For intersecting pairs, distinguish open
    # (unshared endpoints don't co-occur in any static hyperedge) from
    # closed (they do co-occur).
    dj, op, cl = [], [], []
    for i in range(n):
        ai, bi = a[i], b[i]
        for j in range(i + 1, n):
            aj, bj = a[j], b[j]
            sh = (ai == aj) + (ai == bj) + (bi == aj) + (bi == bj)
            if sh == 0:
                dj.append((i, j))
            elif sh == 1:
                shared = ai if (ai == aj or ai == bj) else bi
                u1 = bi if shared == ai else ai
                u2 = bj if shared == aj else aj
                if u2 in coo.get(u1, set()):
                    cl.append((i, j))
                else:
                    op.append((i, j))
            # sh == 2 means the same dyadic edge (excluded by uniqueness)
    dj_s = take(dj, MAX_PAIRS_PER_CLASS)
    op_s = take(op, MAX_PAIRS_PER_CLASS)
    cl_s = take(cl, MAX_PAIRS_PER_CLASS)
    d_dj = emd_for_pairs(times_list, times_list, dj_s)
    d_op = emd_for_pairs(times_list, times_list, op_s) if op_s else np.array([])
    d_cl = emd_for_pairs(times_list, times_list, cl_s) if cl_s else np.array([])
    d_int = (np.concatenate([d_op, d_cl])
             if (len(d_op) + len(d_cl)) else np.array([]))
    return dict(
        name=name,
        n_dj=len(dj), n_op=len(op), n_cl=len(cl),
        d=cohens_d(d_int, d_dj),
        d_open_dj=cohens_d(d_op, d_dj),
        d_closed_dj=cohens_d(d_cl, d_dj),
    )


# ---------------------------------------------------------------------------
# Table-3 (e-h3 cross pairs)
# ---------------------------------------------------------------------------

def run_t3(name):
    """Compute the e-h3 d, d_open-dj, d_closed-dj for one dataset."""
    folder = cv9.dataset_folder(name)
    e2_ids, e2_nodes, e2_times = load_size2(folder, THETA2)
    e3_ids, e3_nodes, e3_times = load_size_k(folder, 3, THETA3)
    n2, n3 = len(e2_ids), len(e3_ids)
    if n2 < 2 or n3 < 2:
        return dict(name=name, n_dj=0, n_op=0, n_cl=0,
                    d=float("nan"), d_open_dj=float("nan"),
                    d_closed_dj=float("nan"))
    e2_set = [set(t) for t in e2_nodes]
    e3_set = [set(t) for t in e3_nodes]
    V_kept = sorted(set().union(*e2_set, *e3_set))
    coo = build_cooccurrence(folder, V_kept)

    dj, op, cl = [], [], []
    for i in range(n2):
        Ei = e2_set[i]
        for j in range(n3):
            Ej = e3_set[j]
            shared = Ei & Ej
            sh = len(shared)
            if sh == 0:
                dj.append((i, j))
            elif sh == 1:
                # Test triangle closure on every cross pair of unshared
                # endpoints (1 from e1, 2 from e2 → up to 2 cross pairs).
                unshared_i = Ei - shared
                unshared_j = Ej - shared
                closed = False
                for u1 in unshared_i:
                    for u2 in unshared_j:
                        if u2 in coo.get(u1, set()):
                            closed = True
                            break
                    if closed:
                        break
                (cl if closed else op).append((i, j))
            # sh == 2 means the size-2 edge sits inside the size-3
            # hyperedge — excluded.
    dj_s = take(dj, MAX_PAIRS_PER_CLASS)
    op_s = take(op, MAX_PAIRS_PER_CLASS)
    cl_s = take(cl, MAX_PAIRS_PER_CLASS)
    d_dj = emd_for_pairs(e2_times, e3_times, dj_s) if dj_s else np.array([])
    d_op = emd_for_pairs(e2_times, e3_times, op_s) if op_s else np.array([])
    d_cl = emd_for_pairs(e2_times, e3_times, cl_s) if cl_s else np.array([])
    d_int = (np.concatenate([d_op, d_cl])
             if (len(d_op) + len(d_cl)) else np.array([]))
    return dict(
        name=name,
        n_dj=len(dj), n_op=len(op), n_cl=len(cl),
        d=cohens_d(d_int, d_dj),
        d_open_dj=cohens_d(d_op, d_dj),
        d_closed_dj=cohens_d(d_cl, d_dj),
    )


# ---------------------------------------------------------------------------
# Table-4 (h3-h3 pairs, 1- vs 2-intersecting)
# ---------------------------------------------------------------------------

def run_t4(name):
    """Compute the h3-h3 d_int1-dj and d_int2-int1 for one dataset."""
    folder = cv9.dataset_folder(name)
    e3_ids, e3_nodes, e3_times = load_size_k(folder, 3, THETA3)
    n = len(e3_ids)
    if n < 2:
        return dict(name=name, n_dj=0, n_int1=0, n_int2=0,
                    d_int1_dj=float("nan"), d_int2_int1=float("nan"))
    sets = [set(t) for t in e3_nodes]

    dj, int1, int2 = [], [], []
    for i in range(n):
        Si = sets[i]
        for j in range(i + 1, n):
            sh = len(Si & sets[j])
            if sh == 0:
                dj.append((i, j))
            elif sh == 1:
                int1.append((i, j))
            elif sh == 2:
                int2.append((i, j))
    dj_s = take(dj, MAX_PAIRS_PER_CLASS)
    i1_s = take(int1, MAX_PAIRS_PER_CLASS)
    i2_s = take(int2, MAX_PAIRS_PER_CLASS)
    d_dj = emd_for_pairs(e3_times, e3_times, dj_s) if dj_s else np.array([])
    d_i1 = emd_for_pairs(e3_times, e3_times, i1_s) if i1_s else np.array([])
    d_i2 = emd_for_pairs(e3_times, e3_times, i2_s) if i2_s else np.array([])
    return dict(
        name=name,
        n_dj=len(dj), n_int1=len(int1), n_int2=len(int2),
        d_int1_dj=cohens_d(d_i1, d_dj),
        d_int2_int1=cohens_d(d_i2, d_i1),
    )


# ---------------------------------------------------------------------------
# Top-level driver and tabular writer
# ---------------------------------------------------------------------------

def write_tex(path, header, body_lines):
    with open(path, "w") as fh:
        fh.write("% generated by code/analysis/sihnkim_emd_full.py\n")
        fh.write(header)
        for ln in body_lines:
            fh.write(ln)
        fh.write("\\hline\n\\end{tabular}\n")
    print(f"Wrote {path}")


def main():
    print("=== EMD reproduction of Tables 2, 3, 4 (Cohen's d only) ===\n")

    print("--- Table 2: size-2 edge pairs ---")
    rows2 = []
    for name in DATASETS_T2:
        t0 = time.time()
        r = run_t2(name)
        rows2.append(r)
        print(f"  {name:28s} d={fmt_d(r['d']):>8s} "
              f"d_open-dj={fmt_d(r['d_open_dj']):>8s} "
              f"d_closed-dj={fmt_d(r['d_closed_dj']):>8s} "
              f"({time.time()-t0:.1f}s, n_dj={r['n_dj']:,}, "
              f"n_op={r['n_op']:,}, n_cl={r['n_cl']:,})")

    print("\n--- Table 3: e-h3 pairs ---")
    rows3 = []
    for name in DATASETS_T3:
        t0 = time.time()
        r = run_t3(name)
        rows3.append(r)
        print(f"  {name:28s} d={fmt_d(r['d']):>8s} "
              f"d_open-dj={fmt_d(r['d_open_dj']):>8s} "
              f"d_closed-dj={fmt_d(r['d_closed_dj']):>8s} "
              f"({time.time()-t0:.1f}s, n_dj={r['n_dj']:,}, "
              f"n_op={r['n_op']:,}, n_cl={r['n_cl']:,})")

    print("\n--- Table 4: h3-h3 pairs ---")
    rows4 = []
    for name in DATASETS_T4:
        t0 = time.time()
        r = run_t4(name)
        rows4.append(r)
        print(f"  {name:28s} d_int1-dj={fmt_d(r['d_int1_dj']):>8s} "
              f"d_int2-int1={fmt_d(r['d_int2_int1']):>8s} "
              f"({time.time()-t0:.1f}s, n_dj={r['n_dj']:,}, "
              f"n_int1={r['n_int1']:,}, n_int2={r['n_int2']:,})")

    # Emit LaTeX tabulars (d only) into out/v9_eec2/.
    out_dir = os.path.join(ROOT, "out", "v9_eec2")
    os.makedirs(out_dir, exist_ok=True)

    body = [f"\\texttt{{{r['name']}}} & {fmt_d(r['d'])} & "
            f"{fmt_d(r['d_open_dj'])} & {fmt_d(r['d_closed_dj'])} \\\\\n"
            for r in rows2]
    write_tex(
        os.path.join(out_dir, "TABLE_emd_repro_T2.tex"),
        ("\\begin{tabular}{l r r r}\n\\hline\n"
         "Dataset & $d$ & $d_{\\rm open-dj}$ & $d_{\\rm closed-dj}$ "
         "\\\\\n\\hline\n"),
        body,
    )

    body = [f"\\texttt{{{r['name']}}} & {fmt_d(r['d'])} & "
            f"{fmt_d(r['d_open_dj'])} & {fmt_d(r['d_closed_dj'])} \\\\\n"
            for r in rows3]
    write_tex(
        os.path.join(out_dir, "TABLE_emd_repro_T3.tex"),
        ("\\begin{tabular}{l r r r}\n\\hline\n"
         "Dataset & $d$ & $d_{\\rm open-dj}$ & $d_{\\rm closed-dj}$ "
         "\\\\\n\\hline\n"),
        body,
    )

    body = [f"\\texttt{{{r['name']}}} & {fmt_d(r['d_int1_dj'])} & "
            f"{fmt_d(r['d_int2_int1'])} \\\\\n"
            for r in rows4]
    write_tex(
        os.path.join(out_dir, "TABLE_emd_repro_T4.tex"),
        ("\\begin{tabular}{l r r}\n\\hline\n"
         "Dataset & $d_{{\\rm int}_1{-}{\\rm dj}}$ & "
         "$d_{{\\rm int}_2{-}{\\rm int}_1}$ \\\\\n\\hline\n"),
        body,
    )


if __name__ == "__main__":
    main()
