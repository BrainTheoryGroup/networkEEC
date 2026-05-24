# Analysis pipeline (`code/analysis/`)

These scripts perform the data analysis for `networkEEC_35.tex`. They
load each dataset from `data/`, compute the EEC (and the alternative
distance-based measures used in Appendix A), and write summary
statistics and per-pair arrays to `out/`. The downstream scripts in
`code/figures_tables/` then turn those `.tsv` and `.npz` files into the
LaTeX tabulars and PDFs that the manuscript reads.

```
analysis/
├── README.md                this file
├── eec_helpers.py           shared loaders + EEC math (cosine, Pearson)
├── compile_v9.py            main pipeline; produces out/v9{,_eec2}/summary*.tsv + .npz
├── compile_v9_eec2.py       thin wrapper: `python3 -c "import compile_v9; compile_v9.main('cosine')"`
├── run_new_datasets.py      incremental refresh for one or two datasets
├── sihnkim_emd_full.py      EMD analysis (Appendix A); reproduces Tables 5--7 d-values
└── vanrossum_test.py        van Rossum spike distance (exploratory; not in the paper)
```

## End-to-end pipeline

From the project root:

```bash
# 1. Convert raw downloads -> standardised edges_list.txt / edges_timings.txt
#    (one-time; the converted files are already in data/networks/<slug>/).
python3 code/data_conversion/convert_college_msg.py
python3 code/data_conversion/convert_copenhagen_calls.py
python3 code/data_conversion/convert_friends_family_call.py
python3 code/data_conversion/convert_ia_reality_call.py
python3 code/data_conversion/convert_social_evolution_call.py

# 2. Run the EEC pipeline twice (Pearson for Appendix, cosine for main text).
python3 code/analysis/compile_v9.py pearson
python3 code/analysis/compile_v9.py cosine

# 3. Sihn-Kim EMD analysis -> Appendix A d-values (Tables 5, 6, 7).
python3 code/analysis/sihnkim_emd_full.py

# 4. (Optional) van Rossum side analysis.
python3 code/analysis/vanrossum_test.py

# 5. Render figures and tables that the manuscript uses.
python3 code/figures_tables/build_table19_v2.py
python3 code/figures_tables/render_tables_eec2.py
python3 code/figures_tables/render_figs_eec2.py
```

## What lands where

| Output                                              | Producer                      | Consumed by `networkEEC_35.tex` |
|-----------------------------------------------------|-------------------------------|---------------------------------|
| `out/TABLE_19.tex`                                  | `build_table19_v2.py`         | Table 1 (per-dataset properties) |
| `out/TABLE_eec2_size2_merged.tex`                   | `render_tables_eec2.py`       | Table 2 (size-2 cosine)          |
| `out/TABLE_eec2_size2x3_merged.tex`                 | `render_tables_eec2.py`       | Table 3 (e-h3 cosine)            |
| `out/TABLE_eec2_size3x3_main.tex`                   | `render_tables_eec2.py`       | Table 4 (h3-h3 cosine)           |
| `out/TABLE_app_size{2,2x3,3x3}_pcc_emd.tex`         | hand-written from `out/v9/summary.tsv` and `out/v9_eec2/TABLE_emd_repro_*.tex` | Tables 5, 6, 7 (appendix) |
| `fig2_v9.pdf`, `fig3_v9.pdf`, `out/fig_3x3_v9.pdf`  | `render_figs_eec2.py`         | Figures 2, 3, 5                  |

The `summary.tsv` files in `out/v9/` and `out/v9_eec2/` carry every
per-dataset, per-pair-type number (n_dj, n_open, n_tri, μ's, p-values,
Cohen's d) — useful for ad hoc inspection or as the source of values
to paste into hand-edited tables.

## What `compile_v9.py` does (in one paragraph)

For each dataset listed in `DELTA`, it loads time-stamped events,
bins them into windows of width Δ, computes the EEC matrix (Pearson
or cosine — Eq. 5 vs Eq. 2 of the paper) between every pair of
frequent edges, classifies each pair by # shared nodes and (for
intersecting pairs) by triangle closure, then writes per-class EEC
arrays to `<slug>_<pair_type>.npz` plus one row of summary statistics
to `summary.tsv`. The same procedure is repeated for size-2 ×
size-2, size-2 × size-3 ("e-h3"), and size-3 × size-3 ("h3-h3")
pair types. See the docstrings inside `compile_v9.py` for the
mathematical correspondences with the paper's equations.

## Reproducibility notes

* All EEC computations use 64-bit floats internally; the saved `.npz`
  arrays are downcast to float32 to halve disk usage.
* Subsampling in `sihnkim_emd_full.py` and `vanrossum_test.py` is
  seeded (np.random.default_rng(20260502)) so that results are
  bit-exactly reproducible.
* The triangle-closure test (intersecting-open vs intersecting-closed)
  uses `eec_helpers.build_cooccurrence`. For temporal hypergraphs we
  only consider co-occurrence in dyadic (size-2) static edges when
  testing closure of a size-2 intersecting pair, per the convention
  introduced in v32 of the manuscript (cf. section 3.1.2).
