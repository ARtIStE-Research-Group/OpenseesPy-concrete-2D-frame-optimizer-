import math

from framekit.data_models import Load, Material
from framekit.input_helpers import (ask_choice, ask_float, ask_int, ask_yes_no, ask_element_ids,
                                   ask_float_def, ask_choice_def, ask_int_def, parse_id_list)
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
# W+ : wind from the left  -> windward = left exterior columns, acting to the right
# W- : wind from the right -> windward = right exterior columns, acting to the left
# Optional leeward load: the same storey values times a coefficient, applied at
# the same time on the exterior columns of the other side, in the SAME direction
# (suction on the leeward face pulls the frame the same way as the pressure).
# ============================================================
def build_storey_wind_column_loads(nodes, elements, geom, storey_values_kNpm, side, direction_sign):
    """Uniform horizontal load on the exterior column(s) of each storey.

    storey_values_kNpm[k] is the load on the column of storey k+1 (storey 1 =
    ground storey) [kN/m, magnitude]; direction_sign = +1 to the right, -1 to the left.
    """
    loads = []
    for story, q in enumerate(storey_values_kNpm, start=1):
        if story > geom.num_floors or abs(q) < 1e-15:
            continue
        for eid in get_exterior_column_ids_by_story(nodes, elements, geom, story, side=side):
            loads.append(Load("uniform", direction_sign * abs(q), element=eid, direction="horizontal"))
    return loads


def triangular_wind_storey_values(geom, top_wind_kNpm):
    """Storey values of the previous triangular profile: value at mid-height of each storey."""
    H = geom.num_floors * geom.floor_height
    if H <= 1e-12:
        return [0.0] * geom.num_floors
    return [abs(top_wind_kNpm) * ((s - 0.5) * geom.floor_height / H) for s in range(1, geom.num_floors + 1)]


def _ask_yes_no_def(prompt, default="n"):
    return ask_choice_def(prompt, ["y", "n"], default=default) == "y"


def _ask_wind_storey_values(case, geom):
    vals = []
    for s in range(1, geom.num_floors + 1):
        where = " (ground storey)" if s == 1 else (" (top storey)" if s == geom.num_floors else "")
        vals.append(ask_float(f"{case} distributed load on the storey {s} column{where} [kN/m] (0 if none): ",
                              min_val=0.0))
    return vals


def _ask_leeward_coefficient(case, leeward_side):
    if not _ask_yes_no_def(f"Apply at the same time on the {leeward_side} (leeward) exterior columns the same {case} "
                           f"loads, in the same direction, multiplied by a coefficient? (y/n) [n]: ", "n"):
        return 0.0
    return ask_float_def("Leeward coefficient (leeward load = coefficient x windward load, e.g. 0.6): ",
                         None, min_val=0.0)


def _print_wind_table(case, geom, windward, coeff, windward_side, leeward_side):
    print(f"\n{case}: storey loads [kN/m]  (windward = {windward_side} columns"
          + (f", leeward = {leeward_side} columns x {coeff:g}" if coeff > 0 else "") + ")")
    for s in range(geom.num_floors, 0, -1):
        q = windward[s - 1]
        line = f"  storey {s:2d}: windward {q:8.2f}"
        if coeff > 0:
            line += f"   leeward {coeff * q:8.2f}"
        print(line)
    tot = sum(windward) * geom.floor_height * (1.0 + coeff)
    print(f"  total horizontal force: {tot:.1f} kN")


def ask_wind_load_cases(nodes, elements, geom):
    """Interactive definition of the wind cases W+ and W-; returns (loads_Wplus, loads_Wminus, info)."""
    print("\n--- Wind load on the exterior columns ---")
    print("W+ = wind from the left:  load on the left exterior columns, acting to the right.")
    print("W- = wind from the right: load on the right exterior columns, acting to the left.")
    print("Wind load profile:")
    print("  1) one value per storey, from the ground storey up (uniform on each storey column)")
    print("  2) triangular: 0 at the base, given value at the top (previous input)")
    mode = ask_choice_def("Choose profile (1/2) [1]: ", ["1", "2"], default="1")

    info = {"mode": "per_storey" if mode == "1" else "triangular"}
    defs = {}
    for case, side, other, sign in (("W+", "left", "right", +1), ("W-", "right", "left", -1)):
        if case == "W-" and mode == "1" and any(abs(v) > 1e-15 for v in defs["W+"][0]):
            if _ask_yes_no_def("Use the same storey loads (and leeward coefficient) for W-, mirrored? (y/n) [y]: ",
                               "y"):
                defs["W-"] = defs["W+"]
                _print_wind_table(case, geom, defs[case][0], defs[case][1], side, other)
                continue
        if mode == "1":
            vals = _ask_wind_storey_values(case, geom)
        else:
            top = ask_float(f"Top wind intensity for {case} [kN/m] (0 if none): ", min_val=0.0)
            vals = triangular_wind_storey_values(geom, top)
        coeff = _ask_leeward_coefficient(case, other) if any(abs(v) > 1e-15 for v in vals) else 0.0
        defs[case] = (vals, coeff)
        if any(abs(v) > 1e-15 for v in vals):
            _print_wind_table(case, geom, vals, coeff, side, other)

    out = {}
    for case, side, other, sign in (("W+", "left", "right", +1), ("W-", "right", "left", -1)):
        vals, coeff = defs[case]
        loads = build_storey_wind_column_loads(nodes, elements, geom, vals, side, sign)
        if coeff > 0:
            loads += build_storey_wind_column_loads(nodes, elements, geom, [coeff * v for v in vals], other, sign)
        out[case] = loads
        info[case] = {"windward_storey_kNpm": list(vals), "leeward_coefficient": coeff}
    return out["W+"], out["W-"], info


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
# LIVE (IMPOSED) LOAD CASES
# Each case has a name, a uniform load and the beams it loads, chosen by
# floor and bay; the cases are combined later by the load groups
# (Linear Add, Range Add, Envelope).
# ============================================================
RESERVED_CASES = ("SW", "G", "Q", "W+", "W-", "E")


def get_beam_id_at(nodes, elements, geom, level, bay):
    """Beam of floor `level` (1..num_floors, roof = num_floors) in bay `bay` (1..num_bays)."""
    beams = get_beam_element_ids_by_level(nodes, elements, geom, level)
    beams = sorted(beams, key=lambda e: min(nodes[elements[e][0]][0], nodes[elements[e][1]][0]))
    return beams[bay - 1] if 1 <= bay <= len(beams) else None


def _clean_case_name(raw, default, taken):
    name = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in raw.strip()) or default
    if name.upper() in RESERVED_CASES:
        name = name + "_live"
    while name in taken:
        name += "b"
    return name


def ask_live_load_cases(nodes, elements, geom, taken=()):
    """Interactive live load cases: {name: [Load, ...]} in input order."""
    print("\n--- Live (imposed) load cases ---")
    print("Each case: a name, a uniform load [kN/m] and the beams it loads (by floor and bay).")
    print("The cases are combined later with Linear Add, Range Add or Envelope.")
    n = ask_int_def("Number of live load cases (0 = none) [0]: ", default=0, min_val=0)
    floors = set(range(1, geom.num_floors + 1))
    bays = set(range(1, geom.num_bays + 1))
    out = {}
    for k in range(1, n + 1):
        default = f"live{k:02d}"
        name = _clean_case_name(input(f"Live case {k} name [{default}]: "), default, set(taken) | set(out))
        loads = []
        while True:
            q = ask_float(f"  {name}: load [kN/m] (magnitude, applied downward): ", min_val=0.0)
            while True:
                fl = parse_id_list(input(f"  {name}: floors (all | 1,3 | 1-3; roof = {geom.num_floors}) [all]: ")
                                   or "all", floors)
                if fl:
                    break
                print(f"  ❌ Choose floors between 1 and {geom.num_floors}.")
            while True:
                by = parse_id_list(input(f"  {name}: bays (all | 1,3 | 2-4) [all]: ") or "all", bays)
                if by:
                    break
                print(f"  ❌ Choose bays between 1 and {geom.num_bays}.")
            n_b = 0
            for lv in fl:
                for b in by:
                    eid = get_beam_id_at(nodes, elements, geom, lv, b)
                    if eid is not None and q > 0:
                        loads.append(Load("uniform", -abs(q), element=eid, direction="vertical"))
                        n_b += 1
            print(f"  {name}: {q:g} kN/m on {n_b} beam(s) (floors {_ranges(fl)}, bays {_ranges(by)})")
            if ask_choice_def(f"  Add more beams to {name} with another load? (y/n) [n]: ", ["y", "n"], "n") != "y":
                break
        out[name] = loads
    return out


def _ranges(ids):
    ids = sorted(ids)
    if not ids:
        return "-"
    parts, a, b = [], ids[0], ids[0]
    for x in ids[1:]:
        if x == b + 1:
            b = x
            continue
        parts.append(f"{a}" if a == b else f"{a}-{b}")
        a = b = x
    parts.append(f"{a}" if a == b else f"{a}-{b}")
    return ",".join(parts)


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
    # Live load cases (combined later by the load groups)
    # --------------------------------------------------------
    cases.update(ask_live_load_cases(nodes, elements, geom, taken=cases.keys()))

    # --------------------------------------------------------
    # Wind loads (per storey or triangular, optional leeward part)
    # --------------------------------------------------------
    cases["W+"], cases["W-"], _wind_info = ask_wind_load_cases(nodes, elements, geom)

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