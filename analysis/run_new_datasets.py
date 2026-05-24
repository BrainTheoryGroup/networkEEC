"""Incremental rerun: recompute the size-2 EEC for one or two named
datasets and patch their rows in `summary.tsv` and the per-dataset
`.npz` arrays in place, without re-processing the other 20 datasets.

Useful when a new dataset is added or when a Δ choice for an existing
dataset is revised. Operates on both `out/v9/` (Pearson) and
`out/v9_eec2/` (cosine).

Usage from the project root:

    python3 code/analysis/run_new_datasets.py            # both branches
    python3 code/analysis/run_new_datasets.py cosine     # only cosine
    python3 code/analysis/run_new_datasets.py pearson    # only Pearson

The list of datasets to refresh, plus their Δ values, is hard-coded
below in NEW_DELTA. Edit that dict, then run.
"""

import os
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

import compile_v9 as cv9


# Datasets to refresh and their Δ in native time units. Used last in
# the v34/v35 cycle to add the friends-family-call and
# social-evolution-call rows.
NEW_DELTA = {
    "friends-family-call":   86400,
    "social-evolution-call": 86400,
}


def run_for(similarity):
    """similarity ∈ {'cosine', 'pearson'} — recompute the named
    datasets under the chosen EEC definition and merge the rows into
    the corresponding summary.tsv."""
    out_dir = cv9.OUT_DIRS[similarity]
    os.makedirs(out_dir, exist_ok=True)
    sum_path = os.path.join(out_dir, "summary.tsv")

    # Read the current summary.tsv (if any), preserving rows for
    # datasets we are not refreshing.
    existing = []
    header = None
    if os.path.exists(sum_path):
        with open(sum_path) as fh:
            header = fh.readline().rstrip("\n")
            for ln in fh:
                existing.append(ln.rstrip("\n"))
    fields = cv9.SUMMARY_2_FIELDS
    if header is None:
        header = "\t".join(fields)

    new_rows = []
    for name, delta in NEW_DELTA.items():
        folder = cv9.dataset_folder(name)
        print(f"[{similarity}] === {name}  Δ={delta} ===", flush=True)
        t0 = time.time()
        out = cv9.categorize_2x2(folder, name, delta, similarity)
        if out is None:
            print("  no data, skipping")
            continue
        r_dj, r_open, r_tri = out
        sd = cv9.stat_pack(name, "2x2", delta, r_dj, r_open, r_tri)
        new_rows.append("\t".join(str(sd[k]) for k in fields))
        cv9.maybe_save_arrays(out_dir, name, "2x2", r_dj, r_open, r_tri)
        print(f"  n_dj={len(r_dj)} n_open={len(r_open)} "
              f"n_tri={len(r_tri)}  ({time.time()-t0:.1f}s)", flush=True)

    # Drop any pre-existing rows for the refreshed datasets, then
    # append the new rows.
    keep = [ln for ln in existing
            if not any(ln.startswith(name + "\t") for name in NEW_DELTA)]
    with open(sum_path, "w") as fh:
        fh.write(header + "\n")
        for ln in keep:
            fh.write(ln + "\n")
        for ln in new_rows:
            fh.write(ln + "\n")
    print(f"[{similarity}] Wrote {sum_path} ({len(keep)} kept + "
          f"{len(new_rows)} appended)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_for(sys.argv[1])
    else:
        run_for("cosine")
        run_for("pearson")
