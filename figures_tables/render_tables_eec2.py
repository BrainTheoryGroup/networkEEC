"""Render EEC2 (cosine) tables from out/v9_eec2/.
Reads the v9_eec2 summary files and writes the tables that the manuscript
(networkEEC_NN.tex from v35 onward) reads from out/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_tables_v9 as RT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RT.SUM     = os.path.join(ROOT, "out", "v9_eec2", "summary.tsv")
RT.SUM_3X3 = os.path.join(ROOT, "out", "v9_eec2", "summary_3x3.tsv")


def main():
    rows = RT.load_summary()
    for r in rows["2x2"]:
        r["E_freq"] = str(RT.derive_E_from_pairs_2x2(
            int(r["n_dj"]), int(r["n_ns"]), int(r["n_sh2"])
        ))

    # Write the tables the manuscript references directly into out/, the
    # location v35.tex \input{}s. The merged tables and the size-3x3 main
    # table are the only EEC2 tables cited by v35.
    RT.write_size2_merged(rows["2x2"], "TABLE_eec2_size2_merged.tex")
    RT.write_size2x3_merged(rows["2x3"], "TABLE_eec2_size2x3_merged.tex")

    rows_3x3 = RT.load(RT.SUM_3X3) if os.path.exists(RT.SUM_3X3) else []
    RT.write_3x3_main(rows_3x3, "TABLE_eec2_size3x3_main.tex")


if __name__ == "__main__":
    main()
