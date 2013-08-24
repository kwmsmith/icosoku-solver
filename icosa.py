import operator
from collections import Counter, namedtuple

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

def gen_additive_partitions(n, npart, chunk):
    def _gen_additive_partitions(p, n, npart, chunk, partitions):
        if npart < 0:
            return
        if sum(p) == n:
            partitions.append(p)
            return
        for i in range(p[-1] if p else chunk, 0, -1):
            _gen_additive_partitions(p + [i], n, npart-1, chunk, partitions)
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

def setup_state(pieces, vertices):
    gr = make_vertex_graph(vertices)
    faces = icosa_faces(gr)
    faces_to_pieces = {f:{} for f in faces}
    for piece in pieces:
        for face in faces:
            rots = []
            for rot in range(0, 3):
                rpiece = rot_piece(piece, rot)
                if min(f - p for p, f in zip(rpiece, face)) >= 0:
                    rots.append(rot)
            if rots:
                faces_to_pieces[face][piece] = rots
    pieces_to_faces = {p:{} for p in pieces}
    for face, pcs in faces_to_pieces.items():
        for piece, rots in pcs.items():
            pieces_to_faces[piece][face] = rots
    return faces_to_pieces, pieces_to_faces

def get_bestface(f2p):
    return min((total_piece_options(pcs), face) for face, pcs in f2p.items())[1]

def total_piece_options(pcs):
    return reduce(lambda x,y: x + len(y), pcs.values(), 0)

def order_pieces(pieces, p2f):
    return sorted((total_piece_options(p2f[piece]), piece, rots) for piece, rots in pieces.items())

def search(faces_to_pieces, pieces_to_faces, placements, vtxsum, vtxocc):
    if len(faces_to_pieces) <= 9:
        import ipdb; ipdb.set_trace()
    assert len(faces_to_pieces) + len(placements) == 20
    assert len(faces_to_pieces) == len(pieces_to_faces)

    # check goal, if found, return goal
    if not faces_to_pieces:
        return True

    # bestface = (face with fewest piece options)
    bestface = get_bestface(faces_to_pieces)

    pieces = faces_to_pieces.pop(bestface)

    # order bestface's pieces by total number of placements overall, including rotations.
    ordered_pieces = order_pieces(pieces, pieces_to_faces)

    # bump the vertex occupancy
    for vtx in bestface:
        vtxocc[vtx] += 1
        assert vtxocc[vtx] <= 5

    # try placing each piece in ordered_pieces, checking vertex sums, etc.
    for idx, (rank, piece, rots) in enumerate(ordered_pieces):

        # Detect duplicates
        if idx and piece.value == ordered_pieces[idx-1][1].value:
            continue

        # take piece off the market for other faces...
        other_faces = pieces_to_faces.pop(piece)
        # import ipdb; ipdb.set_trace()
        for oface, _ in other_faces.items():
            try:
                del faces_to_pieces[oface][piece]
            except KeyError:
                pass

        for rot in rots:
            rpiece = rot_piece(piece, rot)
            # update the vertex sums...
            for vtx, points in zip(bestface, rpiece):
                vtxsum[vtx] += points
            # check for vertex inconsistencies
            for vtx in bestface:
                if vtxsum[vtx] > vtx:
                    break
                if (vtx - vtxsum[vtx]) > 3 * (5 - vtxocc[vtx]):
                    break
            else:
                placements[bestface] = rpiece
                if search(faces_to_pieces, pieces_to_faces, placements, vtxsum, vtxocc):
                    return True
                # reset placements
                del placements[bestface]
            # reset vertex sums.
            for vtx, points in zip(bestface, rpiece):
                vtxsum[vtx] -= points

        # piece is back in the running...
        for oface, orots in other_faces.items():
            try:
                faces_to_pieces[oface][piece] = orots
            except KeyError:
                pass
        pieces_to_faces[piece] = other_faces

    # reset vertex occupancy
    for vtx in bestface:
        vtxocc[vtx] -= 1

    # reset faces_to_pieces
    faces_to_pieces[bestface] = pieces

    return False


if __name__ == '__main__':
    gr = make_vertex_graph(vertices)
    ifaces = icosa_faces(gr)
    f2p, p2f = setup_state(pieces, vertices)
    placements = {}
    vtxsum = {v:0 for v in vertices}
    vtxocc = vtxsum.copy()
    search(f2p, p2f, placements, vtxsum, vtxocc)
