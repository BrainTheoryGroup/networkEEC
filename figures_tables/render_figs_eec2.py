"""Render the EEC2 (cosine similarity) figures. Reads npz arrays from
out/v9_eec2/ and writes the figures to the locations that the manuscript
(networkEEC_NN.tex from v35 onward) reads from:
  fig2_v9.pdf       -> repo root
  fig3_v9.pdf       -> repo root
  fig_3x3_v9.pdf    -> out/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_figs_v9 as RF

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RF.V9      = os.path.join(ROOT, "out", "v9_eec2")
RF.SUM     = os.path.join(RF.V9, "summary.tsv")
RF.SUM_3X3 = os.path.join(RF.V9, "summary_3x3.tsv")


# Per-figure output destinations (basename -> absolute path).
FIG_DEST = {
    "fig2_v9.pdf":     os.path.join(ROOT, "fig2_v9.pdf"),
    "fig3_v9.pdf":     os.path.join(ROOT, "fig3_v9.pdf"),
    "fig_3x3_v9.pdf":  os.path.join(ROOT, "out", "fig_3x3_v9.pdf"),
}


import matplotlib.pyplot as plt
_orig_savefig = plt.Figure.savefig


def _route_savefig(self, fname, *a, **kw):
    base = os.path.basename(fname)
    new = FIG_DEST.get(base, fname)
    os.makedirs(os.path.dirname(new), exist_ok=True)
    return _orig_savefig(self, new, *a, **kw)


plt.Figure.savefig = _route_savefig


def main():
    rows = RF.load_summary()
    rows_3x3 = RF.load_summary_3x3()
    RF.fig2_size2(rows["2x2"])
    RF.horizontal_three_class(rows["2x2"], "2x2", "fig3_v9.pdf",
                              "Size-2 hyperedge pairs, three classes")
    RF.horizontal_5class_3x3(rows_3x3, "fig_3x3_v9.pdf")


if __name__ == "__main__":
    main()
