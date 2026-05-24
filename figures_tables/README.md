# Figure and table rendering (`code/figures_tables/`)

These scripts read the analysis outputs from `out/` (the
`summary*.tsv` files and per-dataset `.npz` arrays produced by
`code/analysis/compile_v9.py`) and turn them into the LaTeX
tabulars and PDF figures that `networkEEC_35.tex` consumes.

| Script                    | Inputs                                                                | Outputs                                                                       |
|---------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `build_table19_v2.py`     | `data/networks/<slug>/edges_*.txt`, `data/hypergraphs/<slug>/edges_*.txt` | `out/TABLE_19.tex` (Table 1)                                              |
| `render_tables_v9.py`     | `out/v9/summary.tsv`, `out/v9/summary_3x3.tsv`                         | `out/TABLE_v9_*.tex` (legacy; not referenced by v35)                          |
| `render_tables_eec2.py`   | `out/v9_eec2/summary.tsv`, `out/v9_eec2/summary_3x3.tsv`               | `out/TABLE_eec2_size{2,2x3,3x3}*.tex` (Tables 2, 3, 4)                        |
| `render_figs_v9.py`       | `out/v9/<slug>_<pair>.npz`                                             | `out/fig*_v9.pdf` (legacy; not referenced by v35)                             |
| `render_figs_eec2.py`     | `out/v9_eec2/<slug>_<pair>.npz`                                        | `fig2_v9.pdf`, `fig3_v9.pdf` (root, Figures 2 and 3); `out/fig_3x3_v9.pdf` (Figure 5) |

Run from the project root:

```bash
python3 code/figures_tables/build_table19_v2.py
python3 code/figures_tables/render_tables_eec2.py
python3 code/figures_tables/render_figs_eec2.py
```

These scripts implement plotting and table-formatting logic only and
are not the focus of code review — see `code/analysis/README.md` for
the analysis pipeline.

The appendix tables `out/TABLE_app_size{2,2x3,3x3}_pcc_emd.tex`
(Tables 5, 6, 7) are hand-written rather than auto-generated; their
PCC d-values come from `out/v9/summary.tsv` (Pearson EEC) and their
EMD d-values come from running `code/analysis/sihnkim_emd_full.py`.
