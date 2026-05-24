"""Render the v09 figures from out/v9 outputs:

  out/fig2_v9.pdf   — 9-panel histogram of EEC (ns vs dj) for size-2
                      survivors, with legend only on panel A and no
                      "(n=...)" annotations in the legend.
  out/fig3_v9.pdf   — Horizontal box plot, three-class for size-2.
  out/fig_2x3_v9.pdf — Horizontal box plot, three-class for size-2x3.
  out/fig_3x3_v9.pdf — Horizontal box plot, three-class for size-3x3.
"""

import os
import csv
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
V9 = os.path.join(ROOT, "out", "v9")
SUM = os.path.join(V9, "summary.tsv")
SUM_3X3 = os.path.join(V9, "summary_3x3.tsv")

DJ_NS_THRESH = 20
THREE_OPEN_TRI = 10  # n_open >= 10 AND n_tri >= 10 (n_dj uses DJ_NS_THRESH)

ORDER = [
    "ia-reality-call", "copenhagen-calls",
    "college-msg",
    "friends-family-call", "social-evolution-call",
    "contact-high-school", "contact-primary-school", "DAWN",
    "NDC-classes", "NDC-substances",
    "coauth-DBLP", "coauth-MAG-Geology", "coauth-MAG-History",
    "congress-bills", "email-Enron", "email-Eu",
    "tags-ask-ubuntu", "tags-math-sx", "tags-stack-overflow",
    "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow",
]
ORDER_IDX = {n: i for i, n in enumerate(ORDER)}


def load_summary():
    rows_by_pt = {"2x2": [], "2x3": []}
    with open(SUM) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            pt = r["pair_type"]
            if pt in rows_by_pt:
                rows_by_pt[pt].append(r)
    for pt in rows_by_pt:
        rows_by_pt[pt].sort(key=lambda r: ORDER_IDX.get(r["dataset"], 999))
    return rows_by_pt


def load_summary_3x3():
    rows = []
    if not os.path.exists(SUM_3X3):
        return rows
    with open(SUM_3X3) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
    rows.sort(key=lambda r: ORDER_IDX.get(r["dataset"], 999))
    return rows


def load_arrays(name, pair_type):
    p = os.path.join(V9, f"{name}_{pair_type}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["r_dj"], z["r_open"], z["r_tri"]


def load_arrays_3x3(name):
    p = os.path.join(V9, f"{name}_3x3.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return (z["r_dj"], z["r_ns1_open"], z["r_ns1_tri"],
            z["r_ns2_open"], z["r_ns2_tri"])


def _derive_E_freq(r):
    n_pairs = int(r["n_dj"]) + int(r["n_ns"]) + int(r["n_sh2"])
    if n_pairs == 0:
        return 0
    return int(round((1 + math.sqrt(1 + 8 * n_pairs)) / 2))


def fig2_size2(rows):
    """For size-2 survivors of full power filter
    (E_freq>=50 AND n_dj>=20 AND n_ns>=20), plot a 3x3 grid of EEC
    histograms (ns vs dj)."""
    keep = [r for r in rows if int(r["n_dj"]) >= DJ_NS_THRESH and
            int(r["n_ns"]) >= DJ_NS_THRESH and _derive_E_freq(r) >= 50]
    n = len(keep)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    bins = np.linspace(0.0, 1.0, 41)
    panel = "ABCDEFGHIJKL"
    label_fs = 20  # 2x of previous 10
    title_fs = 20  # 2x of previous 10
    legend_fs = 18  # ~1.2x of v33's 15 (1.25x ideal but kept slightly
                    # below to avoid the legend touching the histograms)
    annot_fs = 15  # ~1.25x of v33's 12 for the per-panel mu/d annotation
    for k, r in enumerate(keep):
        ax = axes[k // ncols, k % ncols]
        arrs = load_arrays(r["dataset"], "2x2")
        if arrs is None:
            ax.set_title(f"({panel[k]}) {r['dataset']} (no data)",
                         fontsize=title_fs)
            continue
        r_dj, r_open, r_tri = arrs
        r_ns = np.concatenate([r_open, r_tri])
        ax.hist(r_dj, bins=bins, density=True, color="#4169E1",
                alpha=0.55, label="dj")
        ax.hist(r_ns, bins=bins, density=True, color="#FF6347",
                alpha=0.55, label="int")
        ax.set_xlim(0.0, 1.0)
        # Integer-style x-ticks: show "0" and "1" at the endpoints rather
        # than "0.0" and "1.0".
        xticks_2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(xticks_2)
        ax.set_xticklabels(
            [("0" if abs(t) < 1e-9 else
              "1" if abs(t - 1.0) < 1e-9 else f"{t:.1f}")
             for t in xticks_2])
        ax.set_xlabel("EEC", fontsize=label_fs)
        if k % ncols == 0:
            ax.set_ylabel("probability density", fontsize=label_fs)
        ax.set_title(f"({panel[k]}) {r['dataset']}", fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=label_fs * 0.6)
        if k == 0:
            ax.legend(loc="lower right", fontsize=legend_fs,
                      frameon=False,
                      bbox_to_anchor=(0.97, 0.05))
        # Annotate means and Cohen's d inside the panel (no p value).
        # Place at top-right, but for DAWN move to top-left to avoid
        # overlap with the saturated histograms in the upper-right corner.
        try:
            d = float(r["d_dj_ns"])
        except (ValueError, TypeError):
            d = float("nan")
        if r["dataset"] == "DAWN":
            x_anchor, ha = 0.05, "left"
        else:
            x_anchor, ha = 0.95, "right"
        ax.text(
            x_anchor, 0.95,
            (f"$\\mu_{{\\rm dj}}={float(r['mean_dj']):.3f}$\n"
             f"$\\mu_{{\\rm int}}={float(r['mean_ns']):.3f}$\n"
             f"$d={d:.2f}$"),
            transform=ax.transAxes, fontsize=annot_fs, family="monospace",
            verticalalignment="top", horizontalalignment=ha,
        )
    # blank unused panels
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.tight_layout()
    out = os.path.join(ROOT, "out", "fig2_v9.pdf")
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"Wrote {out}")


def horizontal_three_class(rows, pair_type, out_name, title):
    def _ok(r):
        if not (int(r["n_dj"]) >= DJ_NS_THRESH and
                int(r["n_open"]) >= THREE_OPEN_TRI and
                int(r["n_tri"]) >= THREE_OPEN_TRI):
            return False
        if pair_type == "2x2" and _derive_E_freq(r) < 50:
            return False
        return True
    keep = [r for r in rows if _ok(r)]
    n = len(keep)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.55 * n + 1.5)))
    cat_colors = {"dj": "#4169E1", "open": "#FF6347", "tri": "#2E8B57"}
    inner_step = 1.0
    group_gap = 1.3
    positions = []
    box_data = []
    box_colors = []
    centers = []
    pos = 0.0
    for r in keep:
        arrs = load_arrays(r["dataset"], pair_type)
        if arrs is None:
            continue
        r_dj, r_open, r_tri = arrs
        for cat, vec in (("dj", r_dj), ("open", r_open), ("tri", r_tri)):
            positions.append(pos)
            box_data.append(vec)
            box_colors.append(cat_colors[cat])
            pos += inner_step
        centers.append(pos - 2 * inner_step)
        pos += group_gap

    bp = ax.boxplot(box_data, positions=positions, widths=0.85,
                    patch_artist=True, showfliers=False, vert=False)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.5)

    ax.set_yticks(centers)
    ax.set_yticklabels([r["dataset"] for r in keep if
                        load_arrays(r["dataset"], pair_type) is not None],
                       fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("EEC", fontsize=15)
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_ylim(positions[-1] + 1, -1)

    # Compute x-axis from the actual whisker reach (5th--95th percentile,
    # since showfliers=False clips at those). If no whisker exceeds 1,
    # cap xmax at 1.0; otherwise set xmax=1.015. xmin tightens to -0.015.
    whisker_max = max(np.percentile(vec, 95) for vec in box_data) \
        if box_data else 1.0
    xmax = 1.0 if whisker_max <= 1.0 else 1.015
    ax.set_xlim(-0.015, xmax)
    # Force integer-style tick labels: 0 and 1 (no 0.0 / 1.0).
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([("0" if abs(t) < 1e-9 else
                         "1" if abs(t - 1.0) < 1e-9 else f"{t:.1f}")
                        for t in xticks])

    handles = [
        Patch(facecolor=cat_colors["dj"], label="dj",
              alpha=0.75, edgecolor="black"),
        Patch(facecolor=cat_colors["open"], label="int$_\\mathrm{open}$",
              alpha=0.75, edgecolor="black"),
        Patch(facecolor=cat_colors["tri"], label="int$_\\mathrm{closed}$",
              alpha=0.75, edgecolor="black"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, "out", out_name)
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"Wrote {out}")


def horizontal_5class_3x3(rows, out_name):
    """Five-class horizontal box plot for 3x3:
    dj / ns1_open / ns1_tri / ns2_open / ns2_tri."""
    keep = [r for r in rows if int(r["n_dj"]) >= DJ_NS_THRESH and
            int(r["n_ns1_open"]) >= THREE_OPEN_TRI and
            int(r["n_ns1_tri"]) >= THREE_OPEN_TRI and
            int(r["n_ns2_open"]) >= THREE_OPEN_TRI and
            int(r["n_ns2_tri"]) >= THREE_OPEN_TRI]
    n = len(keep)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.7 * n + 1.5)))
    cat_colors = {
        "dj":        "#4169E1",
        "ns1_open":  "#FFB347",
        "ns1_tri":   "#FF6347",
        "ns2_open":  "#7AC074",
        "ns2_tri":   "#2E8B57",
    }
    inner_step = 1.0
    group_gap = 1.6
    positions = []
    box_data = []
    box_colors = []
    centers = []
    pos = 0.0
    used_rows = []
    for r in keep:
        arrs = load_arrays_3x3(r["dataset"])
        if arrs is None:
            continue
        used_rows.append(r)
        r_dj, r1o, r1t, r2o, r2t = arrs
        for cat, vec in (("dj", r_dj), ("ns1_open", r1o),
                         ("ns1_tri", r1t), ("ns2_open", r2o),
                         ("ns2_tri", r2t)):
            positions.append(pos)
            box_data.append(vec)
            box_colors.append(cat_colors[cat])
            pos += inner_step
        centers.append(pos - 4 * inner_step + 1.5 * inner_step)
        pos += group_gap

    bp = ax.boxplot(box_data, positions=positions, widths=0.85,
                    patch_artist=True, showfliers=False, vert=False)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.5)

    ax.set_yticks(centers)
    ax.set_yticklabels([r["dataset"] for r in used_rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("EEC")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    if positions:
        ax.set_ylim(positions[-1] + 1, -1)

    handles = [
        Patch(facecolor=cat_colors["dj"], label="dj",
              alpha=0.75, edgecolor="black"),
        Patch(facecolor=cat_colors["ns1_open"],
              label="int$_{1,\\mathrm{open}}$",
              alpha=0.75, edgecolor="black"),
        Patch(facecolor=cat_colors["ns1_tri"],
              label="int$_{1,\\mathrm{closed}}$",
              alpha=0.75, edgecolor="black"),
        Patch(facecolor=cat_colors["ns2_open"],
              label="int$_{2,\\mathrm{open}}$",
              alpha=0.75, edgecolor="black"),
        Patch(facecolor=cat_colors["ns2_tri"],
              label="int$_{2,\\mathrm{closed}}$",
              alpha=0.75, edgecolor="black"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, ncol=1)
    fig.tight_layout()
    out = os.path.join(ROOT, "out", out_name)
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load_summary()
    rows_3x3 = load_summary_3x3()
    fig2_size2(rows["2x2"])
    horizontal_three_class(
        rows["2x2"], "2x2", "fig3_v9.pdf",
        "Size-2 hyperedge pairs, three classes")
    horizontal_three_class(
        rows["2x3"], "2x3", "fig_2x3_v9.pdf",
        "(size-2, size-3) cross hyperedge pairs, three classes")
    horizontal_5class_3x3(rows_3x3, "fig_3x3_v9.pdf")


if __name__ == "__main__":
    main()
