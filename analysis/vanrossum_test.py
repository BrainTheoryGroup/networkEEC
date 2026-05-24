"""Van Rossum spike-train distance — exploratory side analysis.

This is the script that produced the van Rossum diagnostics reported
in the running discussion of the EEC paper (it is *not* used for any
table in `networkEEC_35.tex`, but is retained so that the user can
re-run the comparison cited in the discussion).

Closed form (van Rossum 2001): for two spike trains f, g with
timestamps {t^f_i}, {t^g_j} and exponential kernel exp(-(t-t_i)/t_c)
for t > t_i, the squared distance is

    D^2(f, g) = (1/2) [ S_ff + S_gg - 2 S_fg ]

with  S_xy = sum_{i, j} exp(-|t^x_i - t^y_j| / t_c).

We choose t_c = Δ (the EEC window width for the same dataset). Sign
convention: D^2 is a *distance*, so a smaller value means more
concurrent. Cohen's d below is reported with the (intersecting -
disjoint) sign convention as elsewhere; thus d < 0 means intersecting
pairs are more concurrent.

We subsample at most MAX_PAIRS_PER_CLASS pairs per class per dataset
to keep the runtime tractable on the largest datasets.

Run from the project root:
    python3 code/analysis/vanrossum_test.py
    python3 code/analysis/vanrossum_test.py --only ia-reality-call
"""

import os
import sys
import time
import argparse
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from eec_helpers import load_size2
import compile_v9 as cv9


ROOT = os.path.join(CODE_DIR, "..", "..")

# Subset of size-2 datasets that survive the dj-vs-int filter at
# θ=50 (E_freq ≥ 50, n_dj ≥ 20, n_int ≥ 20). Kept verbatim from the
# version in the running discussion.
DATASETS = [
    ("ia-reality-call",         50, 3600),
    ("contact-high-school",     50,   90),
    ("contact-primary-school",  50,   30),
    ("DAWN",                    50,    1),
    ("congress-bills",          50,   60),
    ("email-Eu",                50, 86400),
    ("tags-ask-ubuntu",         50,  400),
    ("tags-math-sx",            50,  325),
    ("tags-stack-overflow",     50,  400),
]

MAX_PAIRS_PER_CLASS = 5000   # subsample cap per class per dataset
RNG = np.random.default_rng(20260502)


# ---------------------------------------------------------------------------
# Van Rossum kernel pieces
# ---------------------------------------------------------------------------

def vr_self(t, tc):
    """S_self = sum_{i,j} exp(-|t_i - t_j| / tc) over all (i,j) pairs.

    O(N) recurrence on sorted timestamps:
        u_j := sum_{i < j} exp(-(t_j - t_i)/tc)
             = exp(-(t_j - t_{j-1})/tc) * (1 + u_{j-1}).
    Then S_self = N + 2 * sum_j u_j (the diagonal contributes N).
    """
    N = len(t)
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0
    t = np.sort(np.asarray(t, dtype=np.float64))
    diffs = np.diff(t) / tc
    decay = np.exp(-diffs)
    u = 0.0
    s = 0.0
    for j in range(1, N):
        u = decay[j - 1] * (1.0 + u)
        s += u
    return N + 2.0 * s


def vr_cross(tx, ty, tc):
    """S_cross = sum_i sum_j exp(-|tx_i - ty_j|/tc).

    O(N*M) but with a vectorised inner loop. Loops over the shorter
    of the two arrays for cache friendliness.
    """
    if len(tx) == 0 or len(ty) == 0:
        return 0.0
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    if len(tx) > len(ty):
        tx, ty = ty, tx
    s = 0.0
    inv_tc = 1.0 / tc
    for ti in tx:
        s += np.exp(-np.abs(ty - ti) * inv_tc).sum()
    return s


def vr_d2(tx, ty, tc, self_x=None, self_y=None):
    """Squared van Rossum distance D^2 = (S_ff + S_gg - 2 S_fg) / 2."""
    sxx = self_x if self_x is not None else vr_self(tx, tc)
    syy = self_y if self_y is not None else vr_self(ty, tc)
    sxy = vr_cross(tx, ty, tc)
    return 0.5 * (sxx + syy - 2.0 * sxy)


# ---------------------------------------------------------------------------
# Pair classification (size-2 only) and per-dataset driver
# ---------------------------------------------------------------------------

def shared_count_pair(a_i, b_i, a_j, b_j):
    s = 0
    if a_i == a_j or a_i == b_j:
        s += 1
    if b_i == a_j or b_i == b_j:
        s += 1
    return s


def run_dataset(name, theta, delta):
    """Return one row dict of van Rossum stats for one dataset."""
    folder = cv9.dataset_folder(name)
    t0 = time.time()
    eids, nodes_list, times_list = load_size2(folder, theta)
    n = len(eids)
    if n < 2:
        return None
    a = np.array([nl[0] for nl in nodes_list], dtype=np.int64)
    b = np.array([nl[1] for nl in nodes_list], dtype=np.int64)

    tc = float(delta)

    # Pre-compute the per-edge self-similarity term once.
    self_terms = np.empty(n, dtype=np.float64)
    for i in range(n):
        self_terms[i] = vr_self(times_list[i], tc)

    # Enumerate every unordered pair and classify by # shared nodes.
    dj_pairs, int_pairs = [], []
    for i in range(n):
        ai, bi = a[i], b[i]
        for j in range(i + 1, n):
            sh = shared_count_pair(ai, bi, a[j], b[j])
            if sh == 0:
                dj_pairs.append((i, j))
            elif sh == 1:
                int_pairs.append((i, j))
            # sh == 2 would mean duplicate edge; not possible here.

    def take(lst, k):
        if len(lst) <= k:
            return list(lst)
        idx = RNG.choice(len(lst), size=k, replace=False)
        return [lst[i] for i in idx]
    dj_sample = take(dj_pairs, MAX_PAIRS_PER_CLASS)
    int_sample = take(int_pairs, MAX_PAIRS_PER_CLASS)

    def compute_d2_list(pairs):
        out = np.empty(len(pairs), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            out[k] = vr_d2(times_list[i], times_list[j], tc,
                           self_x=self_terms[i], self_y=self_terms[j])
        return out

    d2_dj = compute_d2_list(dj_sample)
    d2_int = compute_d2_list(int_sample)

    mu_dj = float(d2_dj.mean()) if len(d2_dj) else float("nan")
    mu_int = float(d2_int.mean()) if len(d2_int) else float("nan")
    sd_dj = float(d2_dj.std(ddof=1)) if len(d2_dj) > 1 else float("nan")
    sd_int = float(d2_int.std(ddof=1)) if len(d2_int) > 1 else float("nan")

    if (sd_dj == sd_dj) and (sd_int == sd_int) and (sd_dj + sd_int > 0):
        pooled = np.sqrt(0.5 * (sd_dj ** 2 + sd_int ** 2))
        d_eff = (mu_int - mu_dj) / pooled if pooled > 0 else float("nan")
    else:
        d_eff = float("nan")

    elapsed = time.time() - t0
    return dict(
        dataset=name, delta=delta, tc=tc,
        n_dj=len(dj_pairs), n_int=len(int_pairs),
        n_dj_used=len(dj_sample), n_int_used=len(int_sample),
        mu_dj=mu_dj, mu_int=mu_int, d=d_eff,
        elapsed_s=elapsed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated dataset slugs to restrict to")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    rows = []
    for name, theta, delta in DATASETS:
        if only and name not in only:
            continue
        print(f"==> {name}: theta={theta}, Delta={delta}", flush=True)
        try:
            r = run_dataset(name, theta, delta)
        except Exception as e:
            print(f"   error: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"   n_dj={r['n_dj']:>10,} (used {r['n_dj_used']:>5,})  "
              f"n_int={r['n_int']:>8,} (used {r['n_int_used']:>5,})  "
              f"mu_dj={r['mu_dj']:.4g}  mu_int={r['mu_int']:.4g}  "
              f"d={r['d']:+.2f}  ({r['elapsed_s']:.1f}s)", flush=True)

    print("\n\n=== Van Rossum table (D^2; smaller = more concurrent) ===")
    print(f"{'Dataset':28s} {'Delta':>8s} {'t_c':>8s} "
          f"{'mu_dj':>14s} {'mu_int':>14s} {'d':>6s}")
    for r in rows:
        print(f"{r['dataset']:28s} {r['delta']:>8} {r['tc']:>8.0f} "
              f"{r['mu_dj']:>14.4f} {r['mu_int']:>14.4f} {r['d']:>+6.2f}")

    out_tex = os.path.join(ROOT, "out", "v9_eec2", "TABLE_vanrossum_test.tex")
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as fh:
        fh.write("% generated by code/analysis/vanrossum_test.py\n")
        fh.write("\\begin{tabular}{l r r r r r}\n\\hline\n")
        fh.write("Dataset & $\\Delta$ & $t_c$ & "
                 "$\\mu_{\\rm dj}$ & $\\mu_{\\rm int}$ & $d$ \\\\\n\\hline\n")
        for r in rows:
            d_s = (f"${r['d']:.2f}$" if r['d'] < 0 else f"{r['d']:.2f}") \
                if r['d'] == r['d'] else "--"
            fh.write(
                f"\\texttt{{{r['dataset']}}} & {r['delta']} & "
                f"{int(r['tc'])} & {r['mu_dj']:.4f} & {r['mu_int']:.4f} & "
                f"{d_s} \\\\\n"
            )
        fh.write("\\hline\n\\end{tabular}\n")
    print(f"\nWrote {out_tex}")


if __name__ == "__main__":
    main()
