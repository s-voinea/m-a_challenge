"""Regex backend: RE2 (linear-time automaton) if installed, else stdlib `re`.

Python's `re` backtracks through every alternation branch at every character.
On ~39k comments x ~150 keyword patterns that takes minutes. RE2 runs the same
patterns in about a second. All patterns here are written in the RE2-compatible
subset (no lookarounds or backreferences) and are applied to lower-cased text,
so no IGNORECASE flag is needed.
"""
import re

import numpy as np
import pandas as pd

try:
    import re2 as _engine
    BACKEND = 're2'
except ImportError:  # pragma: no cover
    _engine = re
    BACKEND = 're'

_cache = {}


def compile(pattern):
    rx = _cache.get(pattern)
    if rx is None:
        rx = _cache[pattern] = _engine.compile(pattern)
    return rx


def lower(s):
    """Lower-cased list[str] with NaN -> ''."""
    return s.fillna('').astype(str).str.lower().tolist()


def contains(texts, pattern, index):
    rx = compile(pattern)
    return pd.Series([rx.search(t) is not None for t in texts], index=index, dtype=bool)


def extract(texts, pattern, index, n_groups=1):
    """First match's groups as a DataFrame (NaN where no match)."""
    rx = compile(pattern)
    rows = []
    for t in texts:
        m = rx.search(t)
        rows.append(m.groups() if m else (np.nan,) * n_groups)
    return pd.DataFrame(rows, index=index, columns=range(n_groups))


def sub(texts, pattern, repl=' '):
    rx = compile(pattern)
    return [rx.sub(repl, t) for t in texts]


def count_groups(texts, pattern, n_groups):
    """Count matches per named group `g0..g{n-1}` of one combined pattern -> int array."""
    rx = compile(pattern)
    out = np.zeros((len(texts), n_groups), dtype=np.int32)
    for i, t in enumerate(texts):
        for m in rx.finditer(t):
            out[i, int(m.lastgroup[1:])] += 1
    return out
