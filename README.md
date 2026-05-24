# Code for `networkEEC`

```
code/
├── analysis/           Data analysis. Reads data/, writes summary statistics
│                       and per-pair EEC arrays into out/. Heavily annotated.
│                       This is the part to review.
├── data_conversion/    Convert raw downloads in data/raw/ into the
│                       standardised edges_list.txt / edges_timings.txt
│                       schema under data/networks/.
└── figures_tables/     Read the analysis outputs from out/ and produce the
                        LaTeX tabulars and PDF figures referenced by the
                        manuscript. Plotting / formatting only.
```

End-to-end pipeline (run each from the project root):

```bash
# (one-time) raw downloads -> standardised dataset folders
python3 code/data_conversion/convert_college_msg.py
python3 code/data_conversion/convert_copenhagen_calls.py
python3 code/data_conversion/convert_friends_family_call.py
python3 code/data_conversion/convert_ia_reality_call.py
python3 code/data_conversion/convert_social_evolution_call.py

# main analysis (writes out/v9/* and out/v9_eec2/*)
python3 code/analysis/compile_v9.py pearson
python3 code/analysis/compile_v9.py cosine
python3 code/analysis/sihnkim_emd_full.py

# manuscript-ready figures and tables
python3 code/figures_tables/build_table19_v2.py
python3 code/figures_tables/render_tables_eec2.py
python3 code/figures_tables/render_figs_eec2.py
```

See `analysis/README.md` for a more detailed overview of the
analysis pipeline.
