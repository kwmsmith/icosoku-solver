from itertools import permutations

def icosa_perms():
    everything = set(range(2, 13))
    for i in range(2, 8+1):
        rest0 = range(i+1, 13)
        for level0 in permutations(rest0, 4):
            all_rest = everything - set(level0 + (i,))
            for level1 in permutations(all_rest):
                yield (1, i) + level0 + level1

if __name__ == '__main__':
    configs = tuple(icosa_perms())
    import numpy as np
    configs = np.asarray(configs, dtype=np.uint8)
    np.save('configs', configs)
