from collections import Counter

def _gen_additive_partitions(p, n, npart, chunk, partitions):
    if npart < 0:
        return
    if sum(p) == n:
        partitions.append(p)
        return
    for i in range(p[-1] if p else chunk, 0, -1):
        _gen_additive_partitions(p + [i], n, npart-1, chunk, partitions)

def gen_additive_partitions(n, npart, chunk=3):
    parts = []
    _gen_additive_partitions([], n, npart, chunk, parts)
    # Add zeros to each partition to fill out to npart elements.
    parts = [(p+([0] * (npart-len(p)))) for p in parts]
    return parts

def count_additive_partitions(n, npart, chunk=3):
    assert n >= 0
    if not n:
        return 1
    if not npart:
        return 0
    return sum(count_additive_partitions(n-i, npart-1, i) for i in range(min(n, chunk), 0, -1))

def hist(parts):
    ctr = dict(Counter(reduce(list.__add__, parts, [])))
    norm = sum(ctr.values())
    return {k:float(v)/norm for k,v in ctr.items()}
