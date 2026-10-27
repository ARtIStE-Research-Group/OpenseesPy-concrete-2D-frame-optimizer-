from framekit.data_models import Geometry
from framekit.input_helpers import ask_int, ask_float


# ============================================================
# INPUT
# ============================================================
def ask_geometry():
    nf = ask_int("Number of floors: ", 1, None)
    h  = ask_float("Floor height [m]: ", min_val=0.01, allow_zero=False)
    nb = ask_int("Number of bays: ", 1, None)

    bay_lengths = []
    for i in range(nb):
        bay_lengths.append(
            ask_float(f"Length of bay {i+1} [m]: ", min_val=0.01, allow_zero=False)
        )

    return Geometry(nf, h, nb, bay_lengths)


# ============================================================
# GEOMETRY GENERATION
# ============================================================
def generate_nodes(geom: Geometry):
    nodes = {}
    nid = 1

    for f in range(geom.num_floors + 1):
        y = f * geom.floor_height
        x = 0.0

        for L in geom.bay_lengths:
            nodes[nid] = (x, y)
            nid += 1
            x += L

        nodes[nid] = (x, y)
        nid += 1

    return nodes


def generate_elements(geom: Geometry):
    elems = {}
    eid = 1
    nfp = geom.num_bays + 1

    # Columns
    for f in range(geom.num_floors):
        for b in range(nfp):
            ni = f * nfp + b + 1
            nj = ni + nfp
            elems[eid] = (ni, nj, "column")
            eid += 1

    # Beams
    for f in range(1, geom.num_floors + 1):
        start = f * nfp + 1
        for b in range(geom.num_bays):
            elems[eid] = (start + b, start + b + 1, "beam")
            eid += 1

    return elems


def node_id_at(geom: Geometry, floor: int, grid_x: int):
    nfp = geom.num_bays + 1
    return floor * nfp + grid_x + 1