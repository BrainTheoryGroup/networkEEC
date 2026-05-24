"""Build Table 1 (19 datasets, size-2 slice) per the v09 manuscript spec.

Differences from build_table19.py:
- Frequent-edge definition changed to >= theta (was > theta).
- Two genuinely-pairwise call datasets are listed first.
- T column included; theta column dropped.
- Output is LaTeX tabular code, not an image.
"""

import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PAIRWISE_NETWORKS = {
    "ia-reality-call", "copenhagen-calls", "college-msg",
    "friends-family-call", "social-evolution-call",
}


def dataset_folder(name):
    sub = "networks" if name in PAIRWISE_NETWORKS else "hypergraphs"
    return os.path.join(ROOT, "data", sub, name)

# Per-dataset metadata: time-resolution string, span unit and divisor.
# Per v16 spec, time units in this and the time-span column are
# abbreviated as s, d, q, y (second, day, quarter, year).
META = {
    "ia-reality-call":         dict(res="1 s",  span_unit="d",
                                    span_div=86400),
    "copenhagen-calls":        dict(res="1 s",  span_unit="d",
                                    span_div=86400),
    "college-msg":             dict(res="1 s",  span_unit="d",
                                    span_div=86400),
    "friends-family-call":     dict(res="1 s",  span_unit="d",
                                    span_div=86400),
    "social-evolution-call":   dict(res="1 s",  span_unit="d",
                                    span_div=86400),
    "contact-high-school":     dict(res="20 s", span_unit="d",
                                    span_div=4320),    # 86400/20
    "contact-primary-school":  dict(res="20 s", span_unit="d",
                                    span_div=4320),
    "DAWN":                    dict(res="1 q",  span_unit="y",
                                    span_div=4),
    "NDC-classes":             dict(res="1 d",  span_unit="y",
                                    span_div=365),
    "NDC-substances":          dict(res="1 d",  span_unit="y",
                                    span_div=365),
    "coauth-DBLP":             dict(res="1 y",  span_unit="y",
                                    span_div=1),
    "coauth-MAG-Geology":      dict(res="1 y",  span_unit="y",
                                    span_div=1),
    "coauth-MAG-History":      dict(res="1 y",  span_unit="y",
                                    span_div=1),
    "congress-bills":          dict(res="1 d",  span_unit="y",
                                    span_div=365),
    "email-Enron":             dict(res="1 d",  span_unit="y",
                                    span_div=365),
    "email-Eu":                dict(res="1 s",  span_unit="y",
                                    span_div=86400 * 365),
    "tags-ask-ubuntu":         dict(res="1 h",  span_unit="y",
                                    span_div=24 * 365),
    "tags-math-sx":            dict(res="1 h",  span_unit="y",
                                    span_div=24 * 365),
    "tags-stack-overflow":     dict(res="1 h",  span_unit="y",
                                    span_div=24 * 365),
    "threads-ask-ubuntu":      dict(res="1 h",  span_unit="y",
                                    span_div=24 * 365),
    "threads-math-sx":         dict(res="1 h",  span_unit="y",
                                    span_div=24 * 365),
    "threads-stack-overflow":  dict(res="1 h",  span_unit="y",
                                    span_div=24 * 365),
}

# Order: pairwise call data first, then hypergraph datasets.
ORDER = [
    "ia-reality-call",
    "copenhagen-calls",
    "college-msg",
    "friends-family-call",
    "social-evolution-call",
    "contact-high-school",
    "contact-primary-school",
    "DAWN",
    "NDC-classes",
    "NDC-substances",
    "coauth-DBLP",
    "coauth-MAG-Geology",
    "coauth-MAG-History",
    "congress-bills",
    "email-Enron",
    "email-Eu",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "tags-stack-overflow",
    "threads-ask-ubuntu",
    "threads-math-sx",
    "threads-stack-overflow",
]

THETA = 50


def stats_for(folder, theta):
    """For size-2 hyperedges only. Frequent edge: # events >= theta."""
    list_path = os.path.join(folder, "edges_list.txt")
    timing_path = os.path.join(folder, "edges_timings.txt")

    nodes_of = {}
    with open(list_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != 2:
                continue
            eid = int(parts[0])
            nodes_of[eid] = (int(parts[2]), int(parts[3]))

    nodes_used = set()
    for u, v in nodes_of.values():
        nodes_used.add(u); nodes_used.add(v)
    E_edges = len(nodes_of)
    N_nodes = len(nodes_used)

    freq_pairs = []
    span_max = -1
    with open(timing_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sz = int(parts[1])
            if sz != 2:
                continue
            eid = int(parts[0])
            ts = parts[2:]
            if not ts:
                continue
            last = int(ts[-1])
            if last > span_max:
                span_max = last
            if len(ts) >= theta:    # >= theta, per v09 spec
                freq_pairs.append(nodes_of[eid])

    E_freq = len(freq_pairs)
    if E_freq < 2:
        return N_nodes, E_edges, E_freq, span_max, 0, 0

    node_to_idxs = defaultdict(list)
    for i, (u, v) in enumerate(freq_pairs):
        node_to_idxs[u].append(i)
        node_to_idxs[v].append(i)
    share_counts = defaultdict(int)
    for v, idxs in node_to_idxs.items():
        for ii in range(len(idxs)):
            a = idxs[ii]
            for jj in range(ii + 1, len(idxs)):
                b = idxs[jj]
                share_counts[(a, b)] += 1
    ns = sum(1 for c in share_counts.values() if c >= 1)
    total = E_freq * (E_freq - 1) // 2
    dj = total - ns
    return N_nodes, E_edges, E_freq, span_max, dj, ns


def fmt_T(span_native, span_div, unit):
    val = span_native / span_div
    s = f"{val:.1f}" if val < 10 else f"{val:.0f}"
    return f"{s}~{unit}"


def main():
    rows = []
    for name in ORDER:
        meta = META[name]
        folder = dataset_folder(name)
        N, E, Ef, span, dj, ns = stats_for(folder, THETA)
        rows.append({
            "name": name, "N": N, "E": E, "Ef": Ef,
            "res": meta["res"],
            "T": fmt_T(span, meta["span_div"], meta["span_unit"]),
            "dj": dj, "ns": ns,
        })

    # Plain-text echo (with theta>=)
    print(f"{'dataset':24s} {'N':>10s} {'E':>10s} {'E_freq':>7s} "
          f"{'time res':>10s} {'T':>14s} {'n_dj':>11s} {'n_ns':>9s}")
    for r in rows:
        print(f"{r['name']:24s} {r['N']:>10d} {r['E']:>10d} "
              f"{r['Ef']:>7d} {r['res']:>10s} {r['T']:>14s} "
              f"{r['dj']:>11d} {r['ns']:>9d}")

    # LaTeX tabular
    out_tex = os.path.join(ROOT, "out", "TABLE_19.tex")
    with open(out_tex, "w") as fh:
        fh.write(
            "% generated by code/build_table19_v2.py\n"
            "\\begin{tabular}{l r r r l l r r}\n"
            "\\hline\n"
            "Dataset & $N$ & $E$ & $E_{\\rm freq}$ & "
            "time res. & $T$ & $n_{\\rm dj}$ & $n_{\\rm int}$ \\\\\n"
            "\\hline\n"
        )
        for r in rows:
            fh.write(
                f"\\texttt{{{r['name']}}} & "
                f"{r['N']:,} & {r['E']:,} & {r['Ef']:,} & "
                f"{r['res']} & {r['T']} & "
                f"{r['dj']:,} & {r['ns']:,} \\\\\n"
            )
        fh.write("\\hline\n\\end{tabular}\n")
    print(f"\nWrote {out_tex}")


if __name__ == "__main__":
    main()
