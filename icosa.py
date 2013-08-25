import operator
from collections import Counter, namedtuple
from copy import deepcopy

Piece = namedtuple('Piece', 'id value')

pieces = tuple([Piece(i, x) for i, x in 
        enumerate(sorted([
            (0, 3, 3), (0, 0, 2), (2, 2, 2), (1, 2, 3), (0, 1, 2), (0, 1, 1), (1, 2, 3), 
            (0, 2, 1), (1, 3, 2), (0, 2, 1), (0, 0, 1), (0, 2, 2), (3, 3, 3), (0, 0, 0), 
            (1, 3, 2), (0, 2, 1), (0, 1, 2), (1, 1, 1), (0, 0, 3), (0, 1, 2), 
            ]))
        ])

# Vertices are specifed as follows:
#   * The vertex labeled '1' is always first.
#   * The next 5 vertices are attached to '1', listed in CCW order, starting
#   with the least ('2' in this case).
#   * The next 5 vertices are "below" the previous layer, starting with the
#   vertex below and to the left of the first vertex listed in the previous
#   level ('2' in this case).  They are listed in CCW order.
#   * The vertex opposite '1' is listed last.
vertices = [1, 2, 5, 10, 8, 6, 11, 12, 4, 3, 7, 9]

def verify_vertices(vtxs):
    if vtxs[0] != 1:
        return False
    if min(vtxs[1:6]) != vtxs[1]:
        return False
    if set(vtxs) != set(range(1,13)):
        return False
    return True

def check_repeats(lol, num):
    return Counter(reduce(operator.add, lol, ())) == {x:num for x in range(1, 13)}

def make_vertex_graph(vtxs):
    '''
    Takes a list of vertices, returns a dictionary of vertex -> list(vertex)
    connections.  The list of vertices is in CW order.

    '''

    if not verify_vertices(vtxs):
        raise ValueError("Vertices not verified!")

    graph = {}

    first, last = vtxs[0], vtxs[-1]
    level1, level2 = vtxs[1:6], vtxs[6:11]

    graph[first] = tuple(reversed(level1))
    graph[last] = tuple(level2)

    for idx, vtx in enumerate(level1):
        next = (idx+1) % len(level1)
        graph[vtx] = tuple([first, level1[next], level2[next], level2[idx], level1[idx-1]])

    for idx, vtx in enumerate(level2):
        next = (idx+1) % len(level1)
        graph[vtx] = tuple([last, level2[idx-1], level1[idx-1], level1[idx], level2[next]])

    # Verify that there are 5 unique vertices in the vertex tuples.
    assert all(len(set(vs)) == 5 for vs in graph.values())

    # Verify that each vertex shows up exactly 5 times in the vertex tuples.
    assert check_repeats(graph.values(), 5)

    return graph

def icosa_faces(vtx_graph):

    ifaces = []
    for vtx, nbrs in sorted(vtx_graph.items()):
        for idx, nbr in enumerate(nbrs):
            next = nbrs[(idx+1) % len(nbrs)]
            if vtx > nbr or vtx > next:
                continue
            ifaces.append((vtx, nbr, next))

    # Check that we have 20 unique faces.
    assert len(set(ifaces)) == 20
    # Check that each vertex is in exactly 5 faces.
    assert check_repeats(ifaces, 5)

    return ifaces

def _gen_additive_partitions(p, n, npart, chunk, partitions):
    if npart < 0:
        return
    if sum(p) == n:
        partitions.append(p)
        return
    for i in range(p[-1] if p else chunk, 0, -1):
        _gen_additive_partitions(p + [i], n, npart-1, chunk, partitions)

def gen_additive_partitions(n, npart, chunk):
    parts = []
    _gen_additive_partitions([], n, npart, chunk, parts)
    return parts

def count_additive_partitions(n, npart, chunk):
    assert n >= 0
    if not n:
        return 1
    if not npart:
        return 0
    return sum(count_additive_partitions(n-i, npart-1, i) for i in range(min(n, chunk), 0, -1))

def rot_piece(piece, r):
    id, val = piece.id, piece.value
    if r == 0:
        return val
    if r == 1:
        return (val[1], val[2], val[0])
    if r == 2:
        return (val[2], val[0], val[1])

def vertex_to_faces(faces):
    v2f = {i:[] for i in range(1, 13)}
    for face in faces:
        for v in face:
            v2f[v].append(face)
    return {v:tuple(f) for v, f in v2f.items()}

def setup_state(pieces, vertices):
    gr = make_vertex_graph(vertices)
    faces = icosa_faces(gr)
    faces_to_pieces = {f:[0,{}] for f in faces}
    for piece in pieces:
        if piece.value[0] == piece.value[1] == piece.value[2]:
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
                faces_to_pieces[face][1][piece] = rots
                faces_to_pieces[face][0] += len(rots)
    pieces_to_face_cnt = {p:0 for p in pieces}
    for face, (cnt, pcs) in faces_to_pieces.items():
        for piece, rots in pcs.items():
            pieces_to_face_cnt[piece] += len(rots)
    return faces_to_pieces, pieces_to_face_cnt, vertex_to_faces(faces)

def get_bestface(f2p):
    def keyfunc(item):
        return item[1][0]
    return min(f2p.items(), key=keyfunc)[0]

def order_pieces(pieces, p2f):
    return sorted((p2f[piece], piece, rots) for piece, rots in pieces.items())

def set_bounds(faces_to_pieces, pieces_to_face_cnt, upper_bound, lower_bound, vtx, faces):
    for face in faces:
        # if face == (6,7,8):
            # import ipdb; ipdb.set_trace()
        try:
            cnt, pieces = faces_to_pieces[face]
        except KeyError:
            continue
        idx = face.index(vtx)
        for piece, rots in pieces.items():
            newrots = []
            for rot in rots:
                pval = rot_piece(piece, rot)[idx]
                if lower_bound <= pval <= upper_bound:
                    newrots.append(rot)
            if newrots:
                pieces[piece] = newrots
            else:
                del pieces[piece]
            delta = len(rots) - len(newrots)
            faces_to_pieces[face][0] -= delta
            pieces_to_face_cnt[piece] -= delta


def search(faces_to_pieces, pieces_to_face_cnt, placements, vtxsum, vtxocc, vtx2faces):
    assert len(faces_to_pieces) + len(placements) == 20
    assert len(faces_to_pieces) == len(pieces_to_face_cnt)

    # check goal, if found, return goal
    if not faces_to_pieces:
        return True

    # bestface = (face with fewest piece options)
    bestface = get_bestface(faces_to_pieces)

    cnt, pieces = faces_to_pieces.pop(bestface)

    # order bestface's pieces by total number of placements overall, including rotations.
    ordered_pieces = order_pieces(pieces, pieces_to_face_cnt)

    # bump the vertex occupancy
    for vtx in bestface:
        vtxocc[vtx] += 1
        assert vtxocc[vtx] <= 5

    faces_to_pieces_orig = faces_to_pieces
    pieces_to_face_cnt_orig = pieces_to_face_cnt

    # try placing each piece in ordered_pieces, checking vertex sums, etc.
    for idx, (rank, piece, rots) in enumerate(ordered_pieces):

        # Detect duplicates
        if idx and piece.value == ordered_pieces[idx-1][1].value:
            continue

        faces_to_pieces = deepcopy(faces_to_pieces_orig)
        pieces_to_face_cnt = pieces_to_face_cnt_orig.copy()

        # take piece off the market for other faces...
        pieces_to_face_cnt.pop(piece)
        for face, (pcnt, pieces) in faces_to_pieces.items():
            if piece in pieces:
                faces_to_pieces[face][0] -= len(pieces.pop(piece))

        for rot in rots:
            rpiece = rot_piece(piece, rot)
            # update the vertex sums...
            for vtx, points in zip(bestface, rpiece):
                vtxsum[vtx] += points
            # bound faces at each vertex.
            for vtx in bestface:
                upper_bound = min(vtx - vtxsum[vtx], 3)
                lower_bound = 0
                if (5-vtxocc[vtx]) and (vtx - vtxsum[vtx]) > 3 * (5 - vtxocc[vtx]):
                    cutoff = float(vtx-vtxsum[vtx]) / (5-vtxocc[vtx])
                    assert cutoff >= 3
                    lower_bound = cutoff if cutoff >= 3 else 0
                if upper_bound < 0 or lower_bound > 3:
                    break
                if lower_bound > upper_bound:
                    break
                set_bounds(faces_to_pieces, pieces_to_face_cnt, upper_bound, lower_bound, vtx, vtx2faces[vtx])
            else:
                # import ipdb; ipdb.set_trace()
                placements[bestface] = rpiece
                if search(faces_to_pieces, pieces_to_face_cnt, placements, vtxsum, vtxocc, vtx2faces):
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
    faces_to_pieces_orig[bestface] = [cnt, pieces]

    return False


if __name__ == '__main__':
    f2p, p2f, v2f = setup_state(pieces, vertices)
    placements = {}
    vtxsum = {v:0 for v in vertices}
    vtxocc = vtxsum.copy()
    # import ipdb; ipdb.set_trace()
    success = search(f2p, p2f, placements, vtxsum, vtxocc, v2f)
