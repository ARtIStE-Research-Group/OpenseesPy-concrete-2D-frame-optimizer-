import math

from framekit.data_models import Load, Material
from framekit.input_helpers import ask_choice, ask_float, ask_int, ask_yes_no, ask_element_ids
from framekit.constants import G_STD


# ============================================================
# BASIC INPUTS
# ============================================================
def ask_structure_type():
    return ask_choice("Structure type (RC / Steel): ", ["rc", "steel"])


def ask_material(name):
    E = ask_float(f"{name} E [MPa]: ", min_val=1.0, allow_zero=False)
    dens = ask_float(f"{name} density [kg/m3]: ", min_val=1.0, allow_zero=False)
    return Material(name, E, dens)


# ============================================================
# LOAD HELPERS
# ============================================================
def get_beam_element_ids_by_level(nodes, elements, geom, level):
    y_level = level * geom.floor_height
    out = []

    for eid, (ni, nj, etype) in elements.items():
        if etype != "beam":
            continue
        if abs(nodes[ni][1] - y_level) < 1e-9 and abs(nodes[nj][1] - y_level) < 1e-9:
            out.append(eid)

    return out


def get_column_element_ids_by_story(nodes, elements, geom, story):
    y_bot = (story - 1) * geom.floor_height
    y_top = story * geom.floor_height
    out = []

    for eid, (ni, nj, etype) in elements.items():
        if etype != "column":
            continue
        yi = nodes[ni][1]
        yj = nodes[nj][1]

        if abs(min(yi, yj) - y_bot) < 1e-9 and abs(max(yi, yj) - y_top) < 1e-9:
            out.append(eid)

    return out


def get_exterior_column_ids_by_story(nodes, elements, geom, story, side="left"):
    """
    Return only the exterior columns of one story:
    - left  = column at minimum x
    - right = column at maximum x
    """
    story_cols = get_column_element_ids_by_story(nodes, elements, geom, story)
    if not story_cols:
        return []

    xvals = []
    for eid in story_cols:
        ni, nj, _ = elements[eid]
        xi = nodes[ni][0]
        xj = nodes[nj][0]
        xvals.append((eid, 0.5 * (xi + xj)))

    if side.lower() == "left":
        x_target = min(x for _, x in xvals)
    else:
        x_target = max(x for _, x in xvals)

    tol = 1e-9
    return [eid for eid, x in xvals if abs(x - x_target) < tol]


def _sec_dim_mm(sec, *names):
    """Return the first existing attribute (avoids eager getattr defaults)."""
    for n in names:
        if hasattr(sec, n):
            return float(getattr(sec, n))
    raise AttributeError(f"Section object missing attributes. Tried: {names}")


def compute_self_weight(elements, material, beam_sec, column_sec, element_sections=None):
    """
    Self-weight of frame members as uniform vertical line loads [kN/m],
    applied downward (negative global Y).

    element_sections: optional {eid: section} overriding the default
    beam/column section (e.g. jacketed members), so that weight follows
    the actual section of each element.
    """
    loads = []

    def area_m2(sec):
        b = _sec_dim_mm(sec, "b_mm", "b")
        h = _sec_dim_mm(sec, "h_mm", "h")
        return (b / 1000.0) * (h / 1000.0)

    for eid, (_, _, etype) in elements.items():
        if element_sections is not None and int(eid) in element_sections:
            sec = element_sections[int(eid)]
        else:
            sec = beam_sec if etype == "beam" else column_sec
        w = material.density * G_STD * area_m2(sec) / 1000.0  # kN/m
        loads.append(Load("uniform", -w, element=eid, direction="vertical"))

    return loads


def compute_wall_self_weight(nodes, geom, walls, wall_tw_mm, wall_density_kgm3):
    """
    Self-weight of infill shear-wall panels as vertical point loads [kN].
    Each panel weight (rho*g*t*L*h) is split equally between its four
    corner nodes (half to the top floor, half to the bottom floor), so it
    enters both the gravity analysis and the seismic floor masses.
    """
    from framekit.geometry import node_id_at

    loads = []
    if not walls:
        return loads

    for w in walls:
        tw_m = float(getattr(w, "tw_mm", None) or wall_tw_mm) / 1000.0
        n1 = node_id_at(geom, w.floor_bot, w.bay_left - 1)
        n2 = node_id_at(geom, w.floor_bot, w.bay_right - 1)
        n3 = node_id_at(geom, w.floor_top, w.bay_right - 1)
        n4 = node_id_at(geom, w.floor_top, w.bay_left - 1)

        L = abs(nodes[n2][0] - nodes[n1][0])
        H = abs(nodes[n4][1] - nodes[n1][1])
        W_kN = float(wall_density_kgm3) * G_STD * tw_m * L * H / 1000.0

        for nid in (n1, n2, n3, n4):
            loads.append(Load("point", -0.25 * W_kN, node=nid, direction="vertical"))

    return loads


# ============================================================
# WIND LOAD
# W+ : left exterior columns only, acting to the right
# W- : right exterior columns only, acting to the left
# ============================================================
def build_triangular_wind_column_loads(nodes, elements, geom, top_wind_kNpm, side, direction_sign):
    loads = []
    H = geom.num_floors * geom.floor_height

    if H <= 1e-12 or abs(top_wind_kNpm) < 1e-15:
        return loads

    top_wind_kNpm = abs(top_wind_kNpm)

    for story in range(1, geom.num_floors + 1):
        col_ids = get_exterior_column_ids_by_story(nodes, elements, geom, story, side=side)
        if not col_ids:
            continue

        y_mid = (story - 0.5) * geom.floor_height
        q_story = direction_sign * top_wind_kNpm * (y_mid / H)

        for eid in col_ids:
            loads.append(Load("uniform", q_story, element=eid, direction="horizontal"))

    return loads


# ============================================================
# MAIN LOAD INPUT
# ============================================================
def ask_loads_by_cases(nodes, elements, geom):
    cases = {"G": [], "Q": [], "W+": [], "W-": []}

    # --------------------------------------------------------
    # G loads on floors
    # --------------------------------------------------------
    typical_levels = list(range(1, geom.num_floors)) if geom.num_floors >= 2 else []
    for lv in typical_levels:
        beams = get_beam_element_ids_by_level(nodes, elements, geom, lv)
        if beams:
            q = -abs(ask_float(f"G distributed load on floor {lv} [kN/m] (magnitude, applied downward): "))
            for eid in beams:
                cases["G"].append(Load("uniform", q, element=eid, direction="vertical"))

    roof_beams = get_beam_element_ids_by_level(nodes, elements, geom, geom.num_floors)
    if roof_beams:
        q_roof = -abs(ask_float("G distributed load on roof [kN/m] (magnitude, applied downward): "))
        for eid in roof_beams:
            cases["G"].append(Load("uniform", q_roof, element=eid, direction="vertical"))

    # --------------------------------------------------------
    # Wind loads asked separately
    # --------------------------------------------------------
    print("\n--- Automatic wind load on exterior columns ---")
    print("W+ = left exterior columns only, acting to the right.")
    print("W- = right exterior columns only, acting to the left.")

    top_wind_plus = ask_float("Top wind intensity for W+ [kN/m] (0 if none): ", min_val=0.0)
    if abs(top_wind_plus) > 1e-15:
        cases["W+"] = build_triangular_wind_column_loads(
            nodes, elements, geom,
            top_wind_plus,
            side="left",
            direction_sign=+1
        )

    top_wind_minus = ask_float("Top wind intensity for W- [kN/m] (0 if none): ", min_val=0.0)
    if abs(top_wind_minus) > 1e-15:
        cases["W-"] = build_triangular_wind_column_loads(
            nodes, elements, geom,
            top_wind_minus,
            side="right",
            direction_sign=-1
        )

    # --------------------------------------------------------
    # Extra user loads
    # --------------------------------------------------------
    print("\n--- Extra loads ---")
    print("Choose case: G or Q")
    while ask_yes_no("Add extra load? (y/n): "):
        lc = ask_choice("Case (G/Q): ", ["g", "q"]).upper()
        lt = ask_choice("Type (uniform/point/triangular): ", ["uniform", "point", "triangular"])
        direction = ask_choice("Direction (vertical/horizontal): ", ["vertical", "horizontal"])

        if lt == "point":
            node = ask_int("Node ID: ", 1, max(nodes.keys()))
            while node not in nodes:
                print("❌ Invalid node ID.")
                node = ask_int("Node ID: ", 1, max(nodes.keys()))
            val = ask_float("Point load [kN] (sign): ")
            cases[lc].append(Load("point", val, node=node, direction=direction))

        elif lt == "uniform":
            ids = ask_element_ids("Element IDs (all | 2,3,10-12): ", elements)
            val = ask_float("Uniform load [kN/m] (sign): ")
            for eid in ids:
                cases[lc].append(Load("uniform", val, element=eid, direction=direction))

        elif lt == "triangular":
            ids = ask_element_ids("Element IDs (all | 2,3,10-12): ", elements)
            w1 = ask_float("Triangular start intensity w1 [kN/m] (sign): ")
            w2 = ask_float("Triangular end intensity   w2 [kN/m] (sign): ")
            for eid in ids:
                cases[lc].append(Load("triangular", 0.0, element=eid, direction=direction, w1=w1, w2=w2))

    return cases