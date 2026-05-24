"""Convert copenhagen-calls/calls.csv -> data/networks/copenhagen-calls/ schema.

Input header: timestamp, caller, callee, duration. Timestamp is in seconds
relative to study start (from the calls.README in the figshare archive).

Negative duration (-1) means missed call; we still treat it as an event,
following standard practice.

Symmetrise (a, b) := (min, max). Drop self-loops.
"""

import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.path.join(ROOT, "data", "raw", "copenhagen-calls", "calls.csv")
OUT_DIR = os.path.join(ROOT, "data", "networks", "copenhagen-calls")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    edge_to_times = defaultdict(list)
    n_self = 0
    min_t = None
    max_t = 0

    with open(SRC) as fh:
        next(fh)  # header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                t = int(parts[0])
                u = int(parts[1])
                v = int(parts[2])
            except ValueError:
                continue
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
        fh.write(f"source: Copenhagen Networks Study, calls layer "
                 f"(figshare 7267433)\n")
        fh.write(f"number of edges = {n_edges}\n")
        fh.write(f"number of nodes = {n_nodes}\n")
        fh.write(f"time period = {max_t - min_t}\n")
        fh.write(f"native time unit: 1 second\n")
        fh.write(f"self-loops dropped: {n_self}\n")

    print(f"Wrote {n_edges} unique edges over {n_nodes} nodes; "
          f"time span {(max_t - min_t)/86400:.1f} days. "
          f"Self-loops dropped: {n_self}.")


if __name__ == "__main__":
    main()
