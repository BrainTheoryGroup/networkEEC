"""Cosine-similarity variant of the main pipeline.

Equivalent to running

    python3 code/analysis/compile_v9.py cosine

i.e. it calls `compile_v9.main("cosine")`. The cosine results populate
the main-text figures and tables of `networkEEC_35.tex`; the Pearson
results from `compile_v9.py` populate the appendix tables.

Kept as a separate entry point for backwards compatibility with how the
pipeline used to be wired before v35.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compile_v9


if __name__ == "__main__":
    compile_v9.main("cosine")
