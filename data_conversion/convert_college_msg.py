"""Convert CollegeMsg.txt (SNAP format: src tgt unix_time, space-separated)
into the standard data/networks/<slug>/edges_list.txt / edges_timings.txt schema.

Symmetrise: (u, v) := (min(u,v), max(u,v)).
Drop self-loops.
Shift timestamps so the first event is at t=0.
"""

import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.path.join(ROOT, "data", "raw", "CollegeMsg.txt")
OUT_DIR = os.path.join(ROOT, "data", "networks", "college-msg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    edge_to_times = defaultdict(list)
    n_self = 0
    n_lines = 0
    min_t = None
    max_t = 0

    with open(SRC) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                t = int(parts[2])
            except ValueError:
                continue
            n_lines += 1
            if u == v:
                n_self += 1
                continue
            a, b = (u, v) if u < v else (v, u)
            edge_to_times[(a, b)].append(t)
            if min_t is None or t < min_t:
                min_t = t
            if t > max_t:
                max_t = t

    for k in edge_to_times:
        edge_to_times[k].sort()
    edges = sorted(edge_to_times.keys())
    n_edges = len(edges)
    n_nodes = len(set(n for e in edges for n in e))

    with open(os.path.join(OUT_DIR, "edges_list.txt"), "w") as fh_l, \
         open(os.path.join(OUT_DIR, "edges_timings.txt"), "w") as fh_t:
        for eid, (a, b) in enumerate(edges):
            fh_l.write(f"{eid} 2 {a} {b}\n")
            shifted = [t - min_t for t in edge_to_times[(a, b)]]
            fh_t.write(f"{eid} 2 " + " ".join(str(s) for s in shifted) + "\n")

    with open(os.path.join(OUT_DIR, "info.txt"), "w") as fh:
        fh.write("source: UC-Irvine online messages "
                 "(Opsahl & Panzarasa 2009; SNAP CollegeMsg.txt)\n")
        fh.write(f"raw lines read = {n_lines}\n")
        fh.write(f"number of edges = {n_edges}\n")
        fh.write(f"number of nodes = {n_nodes}\n")
        fh.write(f"time period = {max_t - min_t}\n")
        fh.write("native time unit: 1 second\n")
        fh.write(f"self-loops dropped: {n_self}\n")

    print(f"Wrote {n_edges} edges over {n_nodes} nodes; "
          f"time span {(max_t - min_t)/86400:.1f} days. "
          f"Self-loops dropped: {n_self}.")


if __name__ == "__main__":
    main()
