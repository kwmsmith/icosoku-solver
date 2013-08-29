import operator
from collections import Counter

pieces = tuple(x + (i,) for i, x in 
        enumerate((
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (0, 1, 1),
            (0, 1, 2),
            (0, 1, 2),
            (0, 1, 2),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 2),
            (0, 3, 3),
            (1, 1, 1),
            (1, 2, 3),
            (1, 2, 3),
            (1, 3, 2),
            (1, 3, 2),
            (2, 2, 2),
            (3, 3, 3),
            ))
        )

def verify_vertices(vtxs):
    '''
    Check that the ordered vertex list `vtxs` is valid. Returns True / False
    accordingly.

    Vertices are specifed as follows:
      * The vertex labeled '1' is always first.
      * The next 5 vertices are attached to '1', listed in CCW order, starting
      with the least ('2' in this case).
      * The next 5 vertices are "below" the previous layer, starting with the
      vertex below and to the left of the first vertex listed in the previous
      level ('2' in this case).  They are listed in CCW order.
      * The vertex opposite '1' is listed last.
    For instance:
    vertices = [1, 2, 5, 10, 8, 6, 11, 12, 4, 3, 7, 9]

    '''
    if vtxs[0] != 1:
        return False
    if min(vtxs[1:6]) != vtxs[1]:
        return False
    if set(vtxs) != set(range(1,13)):
        return False
    return True

def check_repeats(lol, num):
    return Counter(reduce(operator.add, lol, ())) == {x:num for x in range(1, 13)}

def make_faces(vertices):
    '''
    Returns all faces given an ordered list of vertices.  Vertices in clockwise
    order.  Smallest-valued vertex is first.
    '''
    if not verify_vertices(vertices):
        raise ValueError("Vertices not verified!")

    first, last = vertices[0], vertices[-1]
    level1, level2 = vertices[1:6], vertices[6:11]

    faces = []

    for idx, v1 in enumerate(level1):
        faces.append((first, level1[(idx+1)%len(level1)], v1))
        faces.append((v1, level2[(idx+1)%len(level2)], level2[idx]))

    for idx, v2 in enumerate(level2):
        faces.append((last, v2, level2[(idx+1)%len(level2)]))
        faces.append((v2, level1[idx-1], level1[idx]))

    def minrot(f):
        minf = min(f)
        if minf == f[0]:
            return f
        elif minf == f[1]:
            return (f[1], f[2], f[0])
        elif minf == f[2]:
            return (f[2], f[0], f[1])
        assert 0

    faces = sorted(minrot(f) for f in faces)

    assert len(set(faces)) == 20
    assert check_repeats(faces, 5)

    return faces

def rot_piece(piece, r):
    if r == 0:
        return (piece[0], piece[1], piece[2])
    if r == 1:
        return (piece[1], piece[2], piece[0])
    if r == 2:
        return (piece[2], piece[0], piece[1])

def vertex_to_faces(faces):
    v2f = {i:[] for i in range(1, 13)}
    for face in faces:
        for v in face:
            v2f[v].append(face)
    return {v:tuple(f) for v, f in v2f.items()}

def setup_state(pieces, vertices):

    faces = make_faces(vertices)

    faces_to_pieces = {f:{} for f in faces}
    for piece in pieces:
        if piece[0] == piece[1] == piece[2]:
            poss_rots = [0]
        else:
            poss_rots = range(3)
        for face in faces:
            rots = []
            for rot in poss_rots:
                rpiece = rot_piece(piece, rot)
                if min(f - p for p, f in zip(rpiece, face)) >= 0:
                    rots.append(rot)
            if rots:
                faces_to_pieces[face][piece] = rots

    return faces_to_pieces, vertex_to_faces(faces)

def get_bestface(f2p):
    def keyfunc(item):
        nrots = 0
        for rots in item[1].values():
            nrots += len(rots)
        return nrots
    return min(f2p.items(), key=keyfunc)[0]

def order_pieces(face, f2p, p2f):
    opieces = [(p2f[piece], piece, rot) for piece, rots in f2p[face][1].items() for rot in rots]
    opieces.sort()
    return opieces

def order_pieces_min_vertex(face, f2p, p2f):

    opieces = []
    for piece, rots in f2p[face][1].items():
        for rot in rots:
            rpiece = rot_piece(piece, rot)
            score = sum(f-p for f,p in zip(face, rpiece))
            assert score >= 0
            opieces.append((score, piece, rot))
    opieces.sort()
    return opieces

def order_pieces_prob(face, pieces, vtxsum, vtxocc):
    from utils import hist
    mul = operator.mul
    hists = {v:hist(v-vtxsum[v], 5-vtxocc[v]) for v in face}
    opieces = []
    for piece, rots in pieces.items():
        for rot in rots:
            rpiece = rot_piece(piece, rot)
            score = reduce(mul, (hists[v][p] for v,p in zip(face, rpiece)), 1.0)
            assert 0 <= score <= 1.0
            opieces.append((score, piece, rpiece))
    opieces.sort(reverse=True)
    return opieces

def set_bounds(faces_to_pieces, upper_bound, lower_bound, vtx, faces):
    for face in faces:
        pieces = faces_to_pieces.get(face, {})
        idx = face.index(vtx)
        for piece, rots in pieces.items():
            newrots = [rot for rot in rots if lower_bound <= piece[(idx+rot)%3] <= upper_bound]
            if newrots:
                pieces[piece] = newrots
            else:
                del pieces[piece]

def copyf2p(f2p):
    return {f:p.copy() for f,p in f2p.items()}

def search(faces_to_pieces, placements, vtxsum, vtxocc, vtx2faces):
    assert len(faces_to_pieces) + len(placements) == 20

    # check goal, if found, return goal
    if not faces_to_pieces:
        return True

    # bestface = (face with fewest piece options)
    bestface = get_bestface(faces_to_pieces)

    pieces_orig = faces_to_pieces.pop(bestface)

    ordered_pieces = order_pieces_prob(bestface, pieces_orig, vtxsum, vtxocc)

    # bump the vertex occupancy
    for vtx in bestface:
        vtxocc[vtx] += 1
        assert vtxocc[vtx] <= 5

    faces_to_pieces_orig = faces_to_pieces

    seen = set()

    # try placing each piece in ordered_pieces, checking vertex sums, etc.
    for rank, piece, rpiece in ordered_pieces:

        if rpiece in seen:
            continue
        seen.add(rpiece)

        faces_to_pieces = copyf2p(faces_to_pieces_orig)

        # take piece off the market for other faces...
        for pieces in faces_to_pieces.values():
            pieces.pop(piece, None)

        # update the vertex sums...
        for vtx, points in zip(bestface, rpiece):
            vtxsum[vtx] += points
        # bound faces at each vertex.
        for vtx in bestface:
            upper_bound = min(vtx - vtxsum[vtx], 3)
            lower_bound = 0
            if (5-vtxocc[vtx]) and (vtx - vtxsum[vtx]) > 3 * (5 - vtxocc[vtx]):
                lower_bound = float(vtx-vtxsum[vtx]) / (5-vtxocc[vtx])
                assert lower_bound >= 3
            if upper_bound < 0 or lower_bound > 3:
                break
            if lower_bound > upper_bound:
                break
            set_bounds(faces_to_pieces, upper_bound, lower_bound, vtx, vtx2faces[vtx])
        else:
            placements[bestface] = (rpiece, piece)
            if search(faces_to_pieces, placements, vtxsum, vtxocc, vtx2faces):
                return True
            # reset placements
            del placements[bestface]
        # reset vertex sums.
        for vtx, points in zip(bestface, rpiece):
            vtxsum[vtx] -= points

    # reset vertex occupancy
    for vtx in bestface:
        vtxocc[vtx] -= 1

    # reset faces_to_pieces
    faces_to_pieces_orig[bestface] = pieces_orig

    return False

def verify_solution(pieces, faces, placements):

    # Check that all faces are included.
    if set(faces) != set(placements.keys()):
        return False, "Not all faces are included in placements."

    # make sure we used all pieces in all slots.
    if set(pieces) != {p for rp, p in placements.values()}:
        return False, "placed pieces not identical to original pieces."

    # Check all vertex sums.
    for vtx, vfaces in vertex_to_faces(faces).items():
        vsum = sum(placements[face][0][face.index(vtx)] for face in vfaces)
        if vsum != vtx:
            return False, "Vertex %d does not have a valid vertex sum (%d)." % (vtx, vsum)

    return True, "All checks out!"

def main(pieces, vertices):
    vertices = list(vertices)
    f2p, v2f = setup_state(pieces, vertices)
    placements = {}
    vtxsum = {v:0 for v in vertices}
    vtxocc = vtxsum.copy()
    success = search(f2p, placements, vtxsum, vtxocc, v2f)
    assert success
    res, msg = verify_solution(pieces, make_faces(vertices), placements)
    assert res
    return {f:rp for f, (rp, p) in placements.items()}

def run(configs):
    for idx, config in enumerate(configs[-1:]):
        print idx, config
        placement = main(pieces, list(config))
        print placement

def run_rand(configs, outfname):
    from random import randrange
    from time import clock
    from cPickle import dump
    seen = set()
    nn = len(configs)
    with open(outfname, 'w') as fh:
        while len(seen) < nn:
            idx = randrange(nn)
            while idx in seen:
                idx = randrange(nn)
            seen.add(idx)
            config = configs[idx]
            t0 = clock()
            placements = main(pieces, config)
            t1 = clock()
            msg = (idx, t1-t0, list(config), placements)
            dump(msg, fh)
            print '{}:{:.2f}:{}:{}'.format(*msg)

def test():
    vertices1 = range(1, 13)
    placements1 = main(pieces, vertices1)
    vertices2 = [1, 2, 5, 10, 8, 6, 11, 12, 4, 3, 7, 9]
    placements2 = main(pieces, vertices2)
    vertices3 = [ 1,  8, 12, 11, 10,  9,  7,  6,  5,  4,  3,  2]
    placements3 = main(pieces, vertices3)
    return placements1, placements2, placements3

if __name__ == '__main__':
    # from numpy import load
    # configs = load('configs.npy')
    # run_rand(configs, 'timings.txt')
    # run(configs)
    p1, p2, p3 = test()
