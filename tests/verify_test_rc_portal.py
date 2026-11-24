"""
Verified example: the 2-storey, 2-bay RC portal of examples/test_rc_portal.

ARCOPY is checked against an independent calculation written in this file with
numpy only (direct stiffness method, no OpenSees, no framekit analysis code):

  1. the console input file builds the model of the description
     (geometry, sections, materials, loads, load groups, seismic mass);
  2. member end forces N, V, M of every basic load case
     (SW, G, live01-04, W+, W-) against the independent solution;
  3. equilibrium: base reactions equal the applied loads of every case;
  4. load groups at result level: CROWD (Range Add) against the 16 explicit
     load patterns, CROWD_SEISM (Linear Add), WIND (Envelope) and two ULS
     combinations, at 41 sections along every member;
  5. seismic masses per floor against the hand calculation;
  6. periods of the two lateral modes against an independent eigen solution.

It also writes the reference tables for the comparison with SAP2000 in
examples/test_rc_portal/reference_results/verification/.

Run (from Spyder or a terminal; it first runs the example, about 1 minute):
    python tests/verify_test_rc_portal.py
    python tests/verify_test_rc_portal.py --run-folder <existing run folder>   (skip the run)
"""
import itertools
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLE = os.path.join(ROOT, "examples", "test_rc_portal")
OUT = os.path.join(EXAMPLE, "reference_results", "verification")
sys.path.insert(0, os.path.join(ROOT, "frame_app"))

RESULTS = []          # (group, name, ok, arcopy, independent)


def check(group, name, arcopy, independent, rtol=1e-6, atol=1e-6):
    a, b = float(arcopy), float(independent)
    ok = abs(a - b) <= max(atol, rtol * abs(b))
    RESULTS.append((group, name, ok, a, b))
    return ok


def check_bool(group, name, ok):
    RESULTS.append((group, name, bool(ok), float(bool(ok)), 1.0))


# =====================================================================
# The structure, as given in the description (Test r.c. portal)
# =====================================================================
G_ACC = 9.81                  # ARCOPY's gravity [m/s2]
E_C = 30e9                    # concrete E [Pa]
E_S = 200e9                   # steel E used by ARCOPY for the transformed sections [Pa]
RHO = 2500.0                  # concrete density [kg/m3] (input of the example)
H_ST, L_BAY, N_ST, N_BAY = 4.0, 5.0, 2, 2

BEAM = dict(b=0.500, h=0.300, cover=0.030, n_top=4, n_bot=4, phi=0.020)       # Φ6/150, 4 legs
COLUMN = dict(b=0.300, h=0.300, cover=0.030, n_top=3, n_bot=3, phi=0.020,     # 8Φ20 = 3+3 + 1+1 side
              n_side=2)                                                       # Φ8/150, 3 legs

G_FLOOR, G_ROOF = 24.0, 20.0                       # kN/m
LIVE = {"live01": (1, 1, 8.0), "live02": (1, 2, 8.0),   # case: (floor, bay, kN/m)
        "live03": (2, 1, 6.0), "live04": (2, 2, 6.0)}
WIND_STOREY = (1.4, 2.8)                           # kN/m on the windward column of storeys 1 and 2
PSI = 0.3

# ARCOPY numbering (geometry.generate_nodes / generate_elements):
#   nodes 1-3 base (x = 0, 5, 10), 4-6 floor 1, 7-9 roof
#   columns 1-3 storey 1 (left, middle, right), 4-6 storey 2; beams 7-8 floor 1, 9-10 roof
NAMES = {1: "C1 storey 1 left", 2: "C2 storey 1 middle", 3: "C3 storey 1 right",
         4: "C4 storey 2 left", 5: "C5 storey 2 middle", 6: "C6 storey 2 right",
         7: "Beam 1 (floor 1, bay 1)", 8: "Beam 2 (floor 1, bay 2)",
         9: "Beam 3 (roof, bay 1)", 10: "Beam 4 (roof, bay 2)"}


def frame_geometry():
    nodes, k = {}, 1
    for st in range(N_ST + 1):
        for ix in range(N_BAY + 1):
            nodes[k] = (ix * L_BAY, st * H_ST)
            k += 1
    nid = lambda st, ix: st * (N_BAY + 1) + ix + 1
    elements, e = {}, 1
    for st in range(1, N_ST + 1):
        for ix in range(N_BAY + 1):
            elements[e] = (nid(st - 1, ix), nid(st, ix), "column")
            e += 1
    for st in range(1, N_ST + 1):
        for ix in range(N_BAY):
            elements[e] = (nid(st, ix), nid(st, ix + 1), "beam")
            e += 1
    return nodes, elements


def section_props(sec, transformed=True):
    """A [m2], I [m4]: gross concrete, plus (transformed=True) the top and bottom
    bars with n = Es/Ec, as in ARCOPY's elastic model."""
    b, h = sec["b"], sec["h"]
    A, I = b * h, b * h ** 3 / 12.0
    if transformed:
        n = E_S / E_C
        a_bar = math.pi * sec["phi"] ** 2 / 4.0
        As = (sec["n_top"] + sec["n_bot"]) * a_bar
        y = h / 2.0 - sec["cover"]
        A += n * As
        I += n * As * y ** 2
    return A, I


def basic_cases(elements):
    """{case: [(element, 'v'|'h', w [kN/m] global)]}, from the description."""
    cases = {}
    cases["SW"] = [(e, "v", -RHO * G_ACC * (BEAM if t == "beam" else COLUMN)["b"]
                    * (BEAM if t == "beam" else COLUMN)["h"] / 1000.0) for e, (_, _, t) in elements.items()]
    cases["G"] = [(7, "v", -G_FLOOR), (8, "v", -G_FLOOR), (9, "v", -G_ROOF), (10, "v", -G_ROOF)]
    beam_at = {(1, 1): 7, (1, 2): 8, (2, 1): 9, (2, 2): 10}
    for name, (fl, bay, w) in LIVE.items():
        cases[name] = [(beam_at[(fl, bay)], "v", -w)]
    # W+: left exterior columns (1 = storey 1, 4 = storey 2), to the right; no leeward part
    cases["W+"] = [(1, "h", WIND_STOREY[0]), (4, "h", WIND_STOREY[1])]
    # W-: mirrored, right exterior columns (3 = storey 1, 6 = storey 2), to the left
    cases["W-"] = [(3, "h", -WIND_STOREY[0]), (6, "h", -WIND_STOREY[1])]
    return cases


# =====================================================================
# Independent direct-stiffness solver (2D Euler-Bernoulli frame)
# =====================================================================
class Frame:
    def __init__(self, nodes, elements, props, i_factor=1.0):
        self.nodes, self.elements, self.props = nodes, elements, props
        self.ids = sorted(nodes)
        self.dof = {n: (3 * k, 3 * k + 1, 3 * k + 2) for k, n in enumerate(self.ids)}
        self.free = [d for n in self.ids if nodes[n][1] > 1e-9 for d in self.dof[n]]
        nd = 3 * len(self.ids)
        self.K = np.zeros((nd, nd))
        self.geo = {}
        for e, (ni, nj, t) in elements.items():
            (x1, y1), (x2, y2) = nodes[ni], nodes[nj]
            L = math.hypot(x2 - x1, y2 - y1)
            c, s = (x2 - x1) / L, (y2 - y1) / L
            A, I = props[t]
            I = I * i_factor
            EA, EI = E_C * A / L, E_C * I
            k = np.array([[EA, 0, 0, -EA, 0, 0],
                          [0, 12 * EI / L ** 3, 6 * EI / L ** 2, 0, -12 * EI / L ** 3, 6 * EI / L ** 2],
                          [0, 6 * EI / L ** 2, 4 * EI / L, 0, -6 * EI / L ** 2, 2 * EI / L],
                          [-EA, 0, 0, EA, 0, 0],
                          [0, -12 * EI / L ** 3, -6 * EI / L ** 2, 0, 12 * EI / L ** 3, -6 * EI / L ** 2],
                          [0, 6 * EI / L ** 2, 2 * EI / L, 0, -6 * EI / L ** 2, 4 * EI / L]])
            T = np.zeros((6, 6))
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            T[:3, :3] = R
            T[3:, 3:] = R
            idx = list(self.dof[ni]) + list(self.dof[nj])
            self.K[np.ix_(idx, idx)] += T.T @ k @ T
            self.geo[e] = (L, c, s, k, T, idx)

    def local_load(self, e, loads):
        """(wx, wy) local uniform load [N/m] on element e."""
        L, c, s = self.geo[e][:3]
        wx = wy = 0.0
        for ee, d, w in loads:
            if ee != e:
                continue
            qx, qy = (w * 1e3, 0.0) if d == "h" else (0.0, w * 1e3)
            wx += c * qx + s * qy
            wy += -s * qx + c * qy
        return wx, wy

    def solve(self, loads):
        """Returns ({e: local end forces [N_i, V_i, M_i, N_j, V_j, M_j] (forces on the element)},
        {base node: reaction (Rx, Ry, Mz)}, U)."""
        F = np.zeros(self.K.shape[0])
        peq = {}
        for e in self.elements:
            L, c, s, k, T, idx = self.geo[e]
            wx, wy = self.local_load(e, loads)
            p = np.array([wx * L / 2, wy * L / 2, wy * L ** 2 / 12, wx * L / 2, wy * L / 2, -wy * L ** 2 / 12])
            peq[e] = p
            F[idx] += T.T @ p
        fr = self.free
        U = np.zeros_like(F)
        U[fr] = np.linalg.solve(self.K[np.ix_(fr, fr)], F[fr])
        forces = {}
        for e in self.elements:
            L, c, s, k, T, idx = self.geo[e]
            forces[e] = k @ (T @ U[idx]) - peq[e]
        R = self.K @ U - F
        reac = {n: tuple(R[list(self.dof[n])]) for n in self.ids if self.nodes[n][1] <= 1e-9}
        return forces, reac, U

    def stations(self, e, f, loads, n=41):
        """N, V, M along element e (tension +, V = dM/dx, M sagging +), from the end forces."""
        L = self.geo[e][0]
        wx, wy = self.local_load(e, loads)
        x = np.linspace(0.0, L, n)
        N = -f[0] - wx * x
        V = f[1] + wy * x
        M = -f[2] + f[1] * x + 0.5 * wy * x ** 2
        return N, V, M

    def lateral_periods(self, floor_mass_kg, n_modes=2):
        """Periods with the floor mass split equally among the floor nodes, horizontal
        DOF only; vertical and rotational DOFs condensed statically."""
        h = [self.dof[nid][0] for nid in self.ids if self.nodes[nid][1] > 1e-9]
        o = [d for d in self.free if d not in h]
        K = self.K
        Kc = K[np.ix_(h, h)] - K[np.ix_(h, o)] @ np.linalg.solve(K[np.ix_(o, o)], K[np.ix_(o, h)])
        m = []
        for nid in self.ids:
            y = self.nodes[nid][1]
            if y > 1e-9:
                st = int(round(y / H_ST))
                m.append(floor_mass_kg[st] / (N_BAY + 1))
        Minv = np.diag(1.0 / np.sqrt(m))
        w2 = np.sort(np.linalg.eigvalsh(Minv @ Kc @ Minv))
        return [2 * math.pi / math.sqrt(v) for v in w2[:n_modes]]


# =====================================================================
# ARCOPY results (OpenSees through framekit)
# =====================================================================
def run_example(tmp):
    """Run main.py with examples/test_rc_portal/inputs.txt in a temporary copy."""
    for f in ("main.py", "arcopy_logo.png", "Logo_PoliTo_dal_2021_blu.png", "artiste_logo_ridotto-768x512.jpg"):
        shutil.copy(os.path.join(ROOT, f), tmp)
    for d in ("frame_app", "records"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    with open(os.path.join(EXAMPLE, "inputs.txt"), encoding="utf-8") as fin:
        p = subprocess.run([sys.executable, "main.py"], stdin=fin, cwd=tmp, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    run = os.path.join(tmp, "test_rc_portal")
    if p.returncode != 0 or not os.path.isdir(os.path.join(run, "_checkpoints")):
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise RuntimeError("the example did not run to the end")
    return run


def load_ck(run, step):
    with open(os.path.join(run, "_checkpoints", f"{step}.pkl"), "rb") as f:
        return pickle.load(f)


def arcopy_case_forces(ck, case_loads):
    import openseespy.opensees as ops
    from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static, \
        ops_ele_end_forces_local
    ops_build_model(ck["nodes"], ck["elements"], E_C, E_S, ck["column_sec"], ck["beam_sec"], geom=ck["geom"])
    ops_apply_combo_loads(ck["nodes"], ck["elements"], case_loads)
    assert ops_run_static() == 0
    ops.reactions()
    f = {e: ops_ele_end_forces_local(e) for e in ck["elements"]}
    r = {n: tuple(ops.nodeReaction(n)) for n, (x, y) in ck["nodes"].items() if abs(y) < 1e-9}
    return f, r


# =====================================================================
# Checks
# =====================================================================
def check_model(ck, ck6, nodes, elements):
    grp = "1 model"
    check_bool(grp, "nodes: 3 x 3 grid, bays 5 m, storeys 4 m", ck["nodes"] == nodes)
    check_bool(grp, "elements: 6 columns + 4 beams, connectivity", ck["elements"] == elements)
    b, c = ck["beam_sec"], ck["column_sec"]
    check_bool(grp, "beam 500 x 300, 4+4 phi20, phi6/150 4 legs",
               (b.b_mm, b.h_mm, b.n_top, b.n_bot, b.phi_top_mm, b.phi_bot_mm, b.stirrup_phi_mm,
                b.stirrup_spacing_mm, b.stirrup_legs) == (500, 300, 4, 4, 20, 20, 6, 150, 4))
    check_bool(grp, "column 300 x 300, 8 phi20 (3+3+1+1), phi8/150 3 legs",
               (c.b_mm, c.h_mm, c.n_top, c.n_bot, c.n_left, c.n_right, c.phi_top_mm, c.phi_side_mm,
                c.stirrup_phi_mm, c.stirrup_spacing_mm, c.stirrup_legs) == (300, 300, 3, 3, 1, 1, 20, 20, 8, 150, 3))
    check(grp, "concrete E [MPa]", ck["material"].E, 30000)
    check(grp, "concrete density [kg/m3]", ck["material"].density, RHO)
    check(grp, "fck [MPa] (C25/30), as given", ck["ec2mat_input"].fck, 25)
    check(grp, "fyk [MPa] (B450C), as given", ck["ec2mat_input"].fyk, 450)
    # EC8-3 existing building, knowledge level KL2: strengths divided by the confidence factor CF = 1.2
    check(grp, "fck used in the checks [MPa] = 25 / CF (KL2, CF = 1.2)", ck["ec2mat"].fck, 25 / 1.2)
    check(grp, "fyk used in the checks [MPa] = 450 / CF (KL2, CF = 1.2)", ck["ec2mat"].fyk, 450 / 1.2)
    ref = basic_cases(elements)
    for case, loads in ref.items():
        got = sorted((L.element, "h" if L.direction == "horizontal" else "v", round(L.value, 9))
                     for L in ck["cases"].get(case, []) if L.load_type == "uniform")
        want = sorted((e, d, round(w, 9)) for e, d, w in loads)
        check_bool(grp, f"loads of case {case}", got == want)
    lg = ck["load_groups"]
    check_bool(grp, "CROWD = Range Add (live01..live04)",
               lg["groups"]["CROWD"] == {"type": "Range Add", "members": list(LIVE)})
    check_bool(grp, "CROWD_SEISM = Linear Add (live01..live04)",
               lg["groups"]["CROWD_SEISM"] == {"type": "Linear Add", "members": list(LIVE)})
    check_bool(grp, "WIND = Envelope (W+; W-)", lg["groups"]["WIND"] == {"type": "Envelope", "members": ["W+", "W-"]})
    check(grp, "psi of the seismic mass", lg["psi_mass"], PSI)
    check_bool(grp, "seismic mass group = CROWD_SEISM, Q = CROWD", (lg["mass"], lg["Q"]) == ("CROWD_SEISM", "CROWD"))


def check_masses(ck6):
    grp = "5 seismic mass"
    sw_b = RHO * G_ACC * BEAM["b"] * BEAM["h"] / 1000.0
    sw_c = RHO * G_ACC * COLUMN["b"] * COLUMN["h"] / 1000.0
    live = {fl: sum(w for (f, _, w) in LIVE.values() if f == fl) * L_BAY for fl in (1, 2)}
    w1 = 2 * L_BAY * sw_b + 2 * L_BAY * G_FLOOR + PSI * live[1] + 3 * H_ST * sw_c       # half of storeys 1 and 2
    w2 = 2 * L_BAY * sw_b + 2 * L_BAY * G_ROOF + PSI * live[2] + 0.5 * 3 * H_ST * sw_c  # half of storey 2
    hand = {1: w1 * 1000 / G_ACC, 2: w2 * 1000 / G_ACC}
    for st in (1, 2):
        check(grp, f"floor {st} mass [kg]", ck6["floor_masses"][st], hand[st], rtol=1e-9)
    return hand


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", default=None, help="existing run folder of the example (skip the run)")
    args = ap.parse_args()

    nodes, elements = frame_geometry()
    tmp = None
    if args.run_folder:
        run = os.path.abspath(args.run_folder)
    else:
        tmp = tempfile.mkdtemp(prefix="arcopy_portal_")
        print("Running the example (examples/test_rc_portal/inputs.txt) ...")
        run = run_example(tmp)
    ck = load_ck(run, "step_02_loads")
    ck6 = load_ck(run, "step_06_modal")

    from framekit.superposition import GroupLoads, envelope
    from framekit.diagrams import _nvm_arrays_for_element

    # ---- 1) model
    check_model(ck, ck6, nodes, elements)

    # ---- 2) + 3) basic cases
    props_t = {"beam": section_props(BEAM), "column": section_props(COLUMN)}
    props_g = {"beam": section_props(BEAM, False), "column": section_props(COLUMN, False)}
    fr_t, fr_g = Frame(nodes, elements, props_t), Frame(nodes, elements, props_g)
    ref_cases = basic_cases(elements)
    rows_t, rows_g, rows_r, per_case = [], [], [], {}
    for case, loads in ref_cases.items():
        f_ind, r_ind, _ = fr_t.solve(loads)
        f_ops, r_ops = arcopy_case_forces(ck, ck["cases"][case])
        scale = max(np.max(np.abs(v)) for v in f_ind.values())
        worst = max(np.max(np.abs(np.asarray(f_ops[e]) - f_ind[e])) for e in elements)
        check("2 basic cases", f"{case}: largest end-force difference / largest end force",
              worst / scale, 0.0, atol=1e-8)
        # equilibrium
        fx = sum(w * 1e3 * fr_t.geo[e][0] for e, d, w in loads if d == "h")
        fy = sum(w * 1e3 * fr_t.geo[e][0] for e, d, w in loads if d == "v")
        check("3 equilibrium", f"{case}: sum of horizontal base reactions [kN]",
              sum(r[0] for r in r_ops.values()) / 1e3, -fx / 1e3, atol=1e-6)
        check("3 equilibrium", f"{case}: sum of vertical base reactions [kN]",
              sum(r[1] for r in r_ops.values()) / 1e3, -fy / 1e3, atol=1e-6)
        f_gr, r_gr, _ = fr_g.solve(loads)
        for e in elements:
            for rows, frm, f in ((rows_t, fr_t, f_ind[e]), (rows_g, fr_g, f_gr[e])):
                N, V, M = frm.stations(e, f, loads, n=2)      # internal forces at the two ends
                for k, end in enumerate(("start", "end")):
                    rows.append({"case": case, "element": e, "member": NAMES[e], "end": end,
                                 "N_kN": N[k] / 1e3, "V_kN": V[k] / 1e3, "M_kNm": M[k] / 1e3})
        for n, r in sorted(r_ops.items()):
            rows_r.append({"case": case, "node": n, "x_m": nodes[n][0], "Rx_kN": r[0] / 1e3,
                           "Ry_kN": r[1] / 1e3, "Mz_kNm": r[2] / 1e3})
        per_case[case] = {e: fr_t.stations(e, f_ind[e], loads) for e in elements}

    # ---- 4) load groups at result level (ARCOPY superposition vs explicit patterns)
    groups = ck["load_groups"]["groups"]
    arc_cases = ck["cases"]

    def arcopy_station_arrays(case):
        import openseespy.opensees as ops  # noqa: F401
        from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static
        ops_build_model(ck["nodes"], ck["elements"], E_C, E_S, ck["column_sec"], ck["beam_sec"], geom=ck["geom"])
        ops_apply_combo_loads(ck["nodes"], ck["elements"], arc_cases[case])
        assert ops_run_static() == 0
        return {e: _nvm_arrays_for_element(e, ck["elements"], arc_cases[case], ck["nodes"], npts=41)
                for e in elements}

    arc_st = {case: arcopy_station_arrays(case) for case in ref_cases}
    combos = {
        "CROWD": [(1.0, "CROWD")],
        "CROWD_SEISM": [(1.0, "CROWD_SEISM")],
        "WIND": [(1.0, "WIND")],
        "ULS_Q_leading_W+": [(1.35, "SW"), (1.35, "G"), (1.5, "CROWD"), (0.9, "W+")],
        "1.35SW+1.35G+1.5CROWD+0.9WIND": [(1.35, "SW"), (1.35, "G"), (1.5, "CROWD"), (0.9, "WIND")],
    }
    live = list(LIVE)
    env_rows = []
    for cname, terms in combos.items():
        gl = GroupLoads((), terms, groups, arc_cases)
        # explicit alternatives: every live pattern (Range Add), every wind direction (Envelope)
        alts = []
        uses_crowd = any(it == "CROWD" for _, it in terms)
        uses_wind = any(it == "WIND" for _, it in terms)
        for pat in (itertools.product([0, 1], repeat=4) if uses_crowd else [None]):
            for wd in (["W+", "W-"] if uses_wind else [None]):
                coef = {}
                for f, it in terms:
                    if it == "CROWD":
                        for on, lc in zip(pat, live):
                            coef[lc] = coef.get(lc, 0.0) + f * on
                    elif it == "CROWD_SEISM":
                        for lc in live:
                            coef[lc] = coef.get(lc, 0.0) + f
                    elif it == "WIND":
                        coef[wd] = coef.get(wd, 0.0) + f
                    else:
                        coef[it] = coef.get(it, 0.0) + f
                alts.append(coef)
        worst = 0.0
        for e in elements:
            for k, comp in ((0, "N"), (1, "V"), (2, "M")):
                hi_a, lo_a = envelope(gl, {c: arc_st[c][e][2 + k] for c in gl.case_names()})
                vals = np.array([sum(cf * per_case[c][e][k] for c, cf in coef.items()) for coef in alts])
                hi_i, lo_i = vals.max(axis=0), vals.min(axis=0)
                sc = max(np.max(np.abs(hi_i)), np.max(np.abs(lo_i)), 1.0)
                worst = max(worst, np.max(np.abs(hi_a - hi_i)) / sc, np.max(np.abs(lo_a - lo_i)) / sc)
                for end, ix in (("start", 0), ("end", -1)):
                    env_rows.append({"combination": cname, "element": e, "member": NAMES[e], "end": end,
                                     "component": comp, "max": hi_i[ix] / 1e3, "min": lo_i[ix] / 1e3})
                if comp == "M" and elements[e][2] == "beam":
                    env_rows.append({"combination": cname, "element": e, "member": NAMES[e], "end": "along member",
                                     "component": "M", "max": hi_i.max() / 1e3, "min": lo_i.min() / 1e3})
        check("4 load groups", f"{cname}: largest max/min difference at 41 sections / largest value "
              f"({len(alts)} explicit alternatives)", worst, 0.0, atol=1e-8)

    # ---- 5) seismic masses
    hand_mass = check_masses(ck6)

    # ---- 6) periods of the lateral modes
    kf = float(ck6["crack_factor"])
    fr_c = Frame(nodes, elements, props_t, i_factor=kf)
    T_ind = fr_c.lateral_periods(hand_mass)
    for m in (0, 1):
        check("6 modal", f"T{m + 1} [s] (EI x {kf:g}, floor mass split over the floor nodes)",
              ck6["periods"][m], T_ind[m], rtol=1e-5)

    # ---- reference tables for SAP2000
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows_t).to_csv(os.path.join(OUT, "member_end_forces_arcopy_model.csv"), index=False,
                                float_format="%.4f")
    pd.DataFrame(rows_g).to_csv(os.path.join(OUT, "member_end_forces_gross_sections.csv"), index=False,
                                float_format="%.4f")
    pd.DataFrame(rows_r).to_csv(os.path.join(OUT, "base_reactions.csv"), index=False, float_format="%.4f")
    pd.DataFrame(env_rows).to_csv(os.path.join(OUT, "group_envelopes.csv"), index=False, float_format="%.4f")
    modal_rows = []
    for label, props, fac in (("ARCOPY model, cracked (EI x %g)" % kf, props_t, kf),
                              ("ARCOPY model, uncracked", props_t, 1.0),
                              ("gross concrete sections, uncracked", props_g, 1.0),
                              ("gross concrete sections, cracked (EI x %g)" % kf, props_g, kf)):
        T = Frame(nodes, elements, props, i_factor=fac).lateral_periods(hand_mass)
        modal_rows.append({"model": label, "T1_s": T[0], "T2_s": T[1]})
    pd.DataFrame(modal_rows).to_csv(os.path.join(OUT, "periods.csv"), index=False, float_format="%.5f")
    pd.DataFrame([{"floor": st, "mass_kg": hand_mass[st], "weight_kN": hand_mass[st] * G_ACC / 1000}
                  for st in (1, 2)]).to_csv(os.path.join(OUT, "floor_masses.csv"), index=False, float_format="%.3f")
    sp = pd.DataFrame([{"member": t, "A_m2": props_t[t][0], "I_m4": props_t[t][1],
                        "A_gross_m2": props_g[t][0], "I_gross_m4": props_g[t][1],
                        "A_modifier_for_SAP": props_t[t][0] / props_g[t][0],
                        "I_modifier_for_SAP": props_t[t][1] / props_g[t][1]} for t in ("beam", "column")])
    sp.to_csv(os.path.join(OUT, "section_properties.csv"), index=False, float_format="%.6g")
    pd.DataFrame([{"group": g, "check": n, "pass": ok, "arcopy": a, "independent": b}
                  for g, n, ok, a, b in RESULTS]).to_csv(os.path.join(OUT, "verification_checks.csv"),
                                                        index=False)

    # ---- summary
    print("\nVerified example: 2-storey, 2-bay RC portal (examples/test_rc_portal)")
    print("ARCOPY (OpenSees via framekit) vs independent direct-stiffness calculation\n")
    last = None
    for g, n, ok, a, b in RESULTS:
        if g != last:
            print(f"--- {g}")
            last = g
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:<78s} {a:14.6g} {b:14.6g}")
    nfail = sum(not r[2] for r in RESULTS)
    print("\nPeriods of the lateral modes:")
    print(pd.DataFrame(modal_rows).to_string(index=False))
    print(f"\nReference tables for SAP2000: {os.path.relpath(OUT, ROOT)}")
    print("\n" + ("ALL CHECKS PASSED" if not nfail else f"{nfail} CHECK(S) FAILED") + f"  ({len(RESULTS)} checks)")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return nfail


if __name__ == "__main__":
    if main():
        sys.exit(1)       # a failed check makes the automatic tests (GitHub) fail
