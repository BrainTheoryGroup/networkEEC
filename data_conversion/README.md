# Data conversion (`code/data_conversion/`)

These five scripts convert raw downloads in `data/raw/` into the
standardised `edges_list.txt` / `edges_timings.txt` / `info.txt`
schema used by every analysis script in `code/analysis/`. They are
one-off scripts (idempotent: rerunning produces the same output).

| Script                              | Reads (under `data/raw/`)                              | Writes (under `data/networks/`)        |
|-------------------------------------|--------------------------------------------------------|----------------------------------------|
| `convert_college_msg.py`            | `CollegeMsg.txt`                                       | `college-msg/`                         |
| `convert_copenhagen_calls.py`       | `copenhagen-calls/calls.csv`                           | `copenhagen-calls/`                    |
| `convert_friends_family_call.py`    | `toSend_realityCommons/FriendsFamily/CallLog.csv`      | `friends-family-call/`                 |
| `convert_ia_reality_call.py`        | `ia-reality-call/ia-reality-call.edges`                | `ia-reality-call/`                     |
| `convert_social_evolution_call.py`  | `toSend_realityCommons/SocialEvolution/Calls.csv`      | `social-evolution-call/`               |

Each script:

1. Reads the raw download.
2. Filters records to those where both endpoints are study participants
   (i.e., not external phone numbers or anonymous identifiers).
3. Drops self-loops.
4. Symmetrises by sorting endpoints — phone calls and one-to-one online
   messages are conventionally treated as undirected interactions.
5. Shifts timestamps so that the earliest event is at t = 0.
6. Writes one row per edge in `edges_list.txt` and `edges_timings.txt`,
   plus a free-form `info.txt` with a one-paragraph provenance note
   and basic counts.

The 17 temporal hypergraphs in `data/hypergraphs/` were already in
this schema when received from the Benson et al. 2018 collection;
no conversion is performed here for those.

Run any of these from the project root, e.g.:

```bash
python3 code/data_conversion/convert_college_msg.py
```
