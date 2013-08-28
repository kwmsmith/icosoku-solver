from operator import add
from collections import Counter


def _gen_additive_partitions(p, n, npart, chunk, partitions):
    if npart < 0:
        return
    if sum(p) == n:
        partitions.append(p)
        return
    for i in range(p[-1] if p else chunk, 0, -1):
        _gen_additive_partitions(p + [i], n, npart-1, chunk, partitions)

def gen_additive_partitions(n, npart, chunk=3, _memo={}):
    try:
        return _memo[n, npart, chunk]
    except (KeyError, TypeError):
        pass
    parts = []
    _gen_additive_partitions([], n, npart, chunk, parts)
    # Add zeros to each partition to fill out to npart elements.
    parts = [(p+([0] * (npart-len(p)))) for p in parts]
    if _memo is not None:
        _memo[n, npart, chunk] = parts
    return parts

def count_additive_partitions(n, npart, chunk=3):
    assert n >= 0
    if not n:
        return 1
    if not npart:
        return 0
    return sum(count_additive_partitions(n-i, npart-1, i) for i in range(min(n, chunk), 0, -1))

def hist(n, npart, chunk=3, _memo={}):
    try:
        return _memo[n, npart, chunk]
    except (KeyError, TypeError):
        pass
    ctr = Counter(reduce(add, gen_additive_partitions(n, npart, chunk), []))
    norm = float(sum(ctr.values()))
    hh = [ctr.get(k, 0.0)/norm for k in range(4)]
    if _memo is not None:
        _memo[n, npart, chunk] = hh
    return hh
