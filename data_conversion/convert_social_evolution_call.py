"""Convert data/raw/toSend_realityCommons/SocialEvolution/Calls.csv into the
standard data/networks/<slug>/edges_list.txt / edges_timings.txt schema.

Header: user_id, time_stamp, duration, dest_user_id_if_known,
dest_phone_hash. We keep only records where dest_user_id_if_known is
non-empty (both endpoints in the study cohort). Calls are conventionally
treated as undirected, so we symmetrize via (a, b) := (min, max) on the
integer user IDs. Self-loops are dropped. Timestamps are converted to
seconds since the first kept event.
"""

import csv
import os
from collections import defaultdict
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.path.join(ROOT, "data", "raw", "toSend_realityCommons", "SocialEvolution",
                   "Calls.csv")
OUT_DIR = os.path.join(ROOT, "data", "networks", "social-evolution-call")


def parse_ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    edge_to_times = defaultdict(list)
    n_self = 0
    n_lines = 0
    n_kept = 0
    min_t = None
    max_t = 0

    with open(SRC, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            n_lines += 1
            a = row.get("user_id", "").strip()
            b = row.get("dest_user_id_if_known", "").strip()
            ts = row.get("time_stamp", "").strip()
            if not a or not b or not ts:
                continue
            try:
                ai = int(a); bi = int(b)
            except ValueError:
                continue
            try:
                t = parse_ts(ts)
            except (ValueError, OverflowError):
                continue
            if ai == bi:
                n_self += 1
                continue
            x, y = (ai, bi) if ai < bi else (bi, ai)
            edge_to_times[(x, y)].append(t)
            if min_t is None or t < min_t:
                min_t = t
            if t > max_t:
                max_t = t
            n_kept += 1

    nodes = sorted(set(n for e in edge_to_times for n in e))
    edges = sorted(edge_to_times.keys())

    with open(os.path.join(OUT_DIR, "edges_list.txt"), "w") as fh_l, \
         open(os.path.join(OUT_DIR, "edges_timings.txt"), "w") as fh_t:
        for eid, (a, b) in enumerate(edges):
            fh_l.write(f"{eid} 2 {a} {b}\n")
            shifted = sorted(t - min_t for t in edge_to_times[(a, b)])
            fh_t.write(f"{eid} 2 " + " ".join(str(s) for s in shifted) +
                       "\n")

    span = max_t - min_t
    with open(os.path.join(OUT_DIR, "info.txt"), "w") as fh:
        fh.write("source: Social Evolution study, calls layer "
                 "(MIT Reality Commons; Calls.csv)\n")
        fh.write(f"raw lines read = {n_lines}\n")
        fh.write(f"records kept (both endpoints in study) = {n_kept}\n")
        fh.write(f"number of edges = {len(edges)}\n")
        fh.write(f"number of nodes = {len(nodes)}\n")
        fh.write(f"time period = {span}\n")
        fh.write("native time unit: 1 second\n")
        fh.write(f"self-loops dropped: {n_self}\n")

    print(f"Wrote {len(edges)} edges over {len(nodes)} nodes; "
          f"time span {span/86400:.1f} days. "
          f"Kept {n_kept}/{n_lines} records.")


if __name__ == "__main__":
    main()
