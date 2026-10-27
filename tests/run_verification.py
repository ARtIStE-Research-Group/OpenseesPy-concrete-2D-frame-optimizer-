"""
ARCOPY verification suite: framekit results vs closed-form hand calculations.
Run from Spyder or a terminal:  python tests/run_verification.py
"""
import os
import sys
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frame_app"))

import openseespy.opensees as ops
from framekit.data_models import Geometry, Material, Load, ShearWall
from framekit.geometry import generate_nodes, generate_elements
from framekit.section_library import RCSection, COLUMN_PRESETS, BEAM_PRESETS
from framekit.loads import compute_self_weight, compute_wall_self_weight
from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static
from framekit.dynamics import build_mass_loadset_from_cases, compute_floor_masses_kg_from_line_loads
from framekit.diagrams import _nvm_arrays_for_element
from framekit.ec2_checks import element_demands_from_local_force
from framekit.response_spectrum import ec8_elastic_spectrum_ordinate, EC8SpectrumParams, cqc_correlation
from framekit.constants import G_STD as g

RESULTS = []


def check(name, value, expected, rtol=1e-3, atol=1e-9):
    ok = abs(value - expected) <= max(atol, rtol * abs(expected))
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:58s} code = {value:12.4f}   hand = {expected:12.4f}")


def section(d, **kw):
    d = dict(d)
    d.pop("label", None)
    d.update(kw)
    return RCSection(**d)


def solve_static(nodes, elems, loads, col, beam, geom, walls=None):
    ops_build_model(nodes, elems, 30e9, 200e9, col, beam, geom=geom, walls=walls)
    ops_apply_combo_loads(nodes, elems, loads)
    assert ops_run_static() == 0
    ops.reactions()


# ------------------------------------------------------------
# 1) Self-weight and seismic masses (3 storeys x 2 bays)
# ------------------------------------------------------------
geom = Geometry(3, 3.0, 2, [5.0, 5.0])
nodes, elems = generate_nodes(geom), generate_elements(geom)
col = section(COLUMN_PRESETS[2], b_mm=400, h_mm=400)
beam = section(BEAM_PRESETS[2], b_mm=300, h_mm=500)
mat = Material("C", 30000, 2500)
w_col, w_beam = 2500 * g * 0.16 / 1000, 2500 * g * 0.15 / 1000
sw = compute_self_weight(elems, mat, beam, col)
solve_static(nodes, elems, sw, col, beam, geom)
base = [n for n, (x, y) in nodes.items() if abs(y) < 1e-9]
check("Self-weight: total vertical base reaction [kN]",
      sum(ops.nodeReaction(n, 2) for n in base) / 1000, 9 * w_col * 3 + 6 * w_beam * 5)

fm = compute_floor_masses_kg_from_line_loads(geom, nodes, elems, build_mass_loadset_from_cases({"SW": sw}))
m_b, m_c = 2 * w_beam * 5 * 1000 / g, 3 * w_col * 3 * 1000 / g
check("Seismic mass floor 1 (beams + columns above/below) [kg]", fm[1], m_b + m_c)
check("Seismic mass roof (beams + half top columns) [kg]", fm[3], m_b + 0.5 * m_c)

wl = compute_wall_self_weight(nodes, geom, [ShearWall(1, 2, 0, 1, tw_mm=200)], 200, 2500)
check("Wall panel self-weight [kN]", -sum(L.value for L in wl), 2500 * g * 0.2 * 5 * 3 / 1000)

# ------------------------------------------------------------
# 2) Single-span beams: loads, N/V/M diagrams
# ------------------------------------------------------------
g1 = Geometry(1, 3.0, 1, [6.0])
n1, e1 = generate_nodes(g1), generate_elements(g1)
stiff = section(COLUMN_PRESETS[2], b_mm=2000, h_mm=2000)   # near-rigid columns -> fixed-end beam
bm = section(BEAM_PRESETS[2], b_mm=300, h_mm=500)
beam_id = [e for e, v in e1.items() if v[2] == "beam"][0]
L, w = 6.0, 20.0
loads = [Load("uniform", -w, element=beam_id, direction="vertical")]
solve_static(n1, e1, loads, stiff, bm, g1)
xg, yg, N, V, M, Lb, c, s = _nvm_arrays_for_element(beam_id, e1, loads, n1, npts=61)
check("Fixed-end beam, UDL: midspan moment (sagging +) [kNm]", M[30] / 1000, w * L**2 / 24, rtol=0.03)
check("Fixed-end beam, UDL: end moment (hogging -) [kNm]", M[0] / 1000, -w * L**2 / 12, rtol=0.03)
check("Fixed-end beam, UDL: end shear [kN]", abs(V[0]) / 1000, w * L / 2, rtol=0.01)

# triangular load really applied (OpenSees -beamLinear used to be ignored)
tri = [Load("triangular", 0.0, element=beam_id, direction="vertical", w1=0.0, w2=-30.0)]
solve_static(n1, e1, tri, stiff, bm, g1)
check("Triangular load 0->30 kN/m: total base reaction [kN]",
      sum(ops.nodeReaction(n, 2) for n in (1, 2)) / 1000, 0.5 * 30 * L)

# ------------------------------------------------------------
# 3) Axial sign: compression positive, tension negative
# ------------------------------------------------------------
NEd_c, *_ = None, None
d_c = element_demands_from_local_force(np.array([100e3, 0, 0, -100e3, 0, 0]))
d_t = element_demands_from_local_force(np.array([-100e3, 0, 0, 100e3, 0, 0]))
check("Signed axial force, compressed member [kN]", float(d_c[-1]), 100.0)
check("Signed axial force, tensioned member [kN]", float(d_t[-1]), -100.0)

# ------------------------------------------------------------
# 4) EC8 spectrum and CQC
# ------------------------------------------------------------
prm = EC8SpectrumParams(ag=0.25 * g, soil_class="B", spectrum_type=1, damping_ratio=0.05)
check("EC8 Type 1, ground B: Se(TB..TC plateau) [m/s2]", ec8_elastic_spectrum_ordinate(0.3, prm), 0.25 * g * 1.2 * 2.5)
check("EC8 Type 1, ground B: Se(T = 1.0 s) [m/s2]", ec8_elastic_spectrum_ordinate(1.0, prm), 0.25 * g * 1.2 * 2.5 * 0.5 / 1.0)
b_, xi = 0.5, 0.05
rho = 8 * xi**2 * (1 + b_) * b_**1.5 / ((1 - b_**2) ** 2 + 4 * xi**2 * b_ * (1 + b_) ** 2)
check("CQC correlation coefficient (beta = 0.5, xi = 5%)", cqc_correlation(b_, xi), rho)

# ------------------------------------------------------------
# 5) EC8 design quantities (ec8_seismic)
# ------------------------------------------------------------
from framekit.ec8_seismic import (design_spectrum_ordinate, compute_behaviour_factor, apply_confidence_factor,
                                  lateral_force_method, evaluate_lateral_force_method, modal_x_properties,
                                  rsa_equivalent_floor_forces, floor_masses_from_combos)
from framekit.data_models import EC2MaterialStrengths
from framekit.combinations import build_eurocode_combinations
from framekit.dynamics import ops_assign_floor_masses
rp = {"ag": 0.25 * g, "soil_class": "B", "spectrum_type": 1, "damping_ratio": 0.05, "direction": 1,
      "importance_factor": 1.0, "combo_method": "CQC"}
check("Design spectrum Sd plateau, q = 3.9 [m/s2]", design_spectrum_ordinate(0.3, rp, 3.9), 0.25 * g * 1.2 * 2.5 / 3.9)
check("Design spectrum lower bound β·ag at T = 3 s [m/s2]", design_spectrum_ordinate(3.0, rp, 3.9), 0.2 * 0.25 * g)
check("q, EN 1998-1 DCM multi-bay multi-storey frame", compute_behaviour_factor(geom, None, "EN1998-1", "DCM", True)[0], 3.0 * 1.3)
check("q, EN 1998-1 DCH frame, not regular in elevation", compute_behaviour_factor(geom, None, "EN1998-1", "DCH", False)[0], 4.5 * 1.3 * 0.8)
mcf = apply_confidence_factor(EC2MaterialStrengths(25.0, 500.0, 1.5, 1.15, 0.85), 1.20)
check("EN 1998-3 KL2: fck = fcm / CF [MPa]", mcf.fck, 25.0 / 1.2)

cases = {"SW": sw, "G": [Load("uniform", -25.0, element=e) for e, v in elems.items() if v[2] == "beam"],
         "Q": [], "W+": [], "W-": []}
combos = build_eurocode_combinations(cases)
st = {"code": "EN1998-1", "rsa_params": rp, "q": 3.9, "nu": 0.5, "dls_limit": 0.005, "beta": 0.2,
      "stiffness_factor": 0.5, "extra_mass_floor_kg": 0.0}
ev = evaluate_lateral_force_method(nodes=nodes, elements=elems, geom=geom, combos=combos, combo_names=list(combos),
                                   Ec_Pa=30e9, Es_Pa=200e9, column_sec=col, beam_sec=beam, settings=st,
                                   ops_build_model=ops_build_model)
M = sum(ev["floor_masses"].values())
check("Lateral force method Fb = Sd(T1)·m·λ [kN]", ev["Fb_kN"], design_spectrum_ordinate(ev["T1"], rp, 3.9) * M * 0.85 / 1000)
z = np.array([3.0, 6.0, 9.0]); mm = np.array([ev["floor_masses"][k] for k in (1, 2, 3)])
check("Lateral force method F3 = Fb·z3·m3/Σz·m [kN]", ev["forces_kN"][3], ev["Fb_kN"] * z[2] * mm[2] / np.sum(z * mm))

# cracked stiffness: T(0.5 EI) / T(EI) ≈ 1/sqrt(0.5) for a frame (axial stiffness unchanged)
masses = floor_masses_from_combos(geom, nodes, elems, combos)
Ts = []
for kf in (1.0, 0.5):
    ops_build_model(nodes, elems, 30e9, 200e9, col, beam, geom=geom, stiffness_factor=kf)
    ops_assign_floor_masses(nodes, geom, masses)
    Ts.append(max(modal_x_properties(geom, 3), key=lambda d: d["ratio"])["T"])
check("Cracked stiffness 0.5 EI: period ratio T_cr / T", Ts[1] / Ts[0], 1.0 / math.sqrt(0.5), rtol=0.03)

# one-storey frame: RSA-equivalent base shear = m·Sd(T1)
c1 = build_eurocode_combinations({"SW": compute_self_weight(e1, mat, bm, stiff), "Q": [], "W+": [], "W-": [],
                                   "G": [Load("uniform", -20.0, element=beam_id)]})
m1 = floor_masses_from_combos(g1, n1, e1, c1)
ops_build_model(n1, e1, 30e9, 200e9, stiff, bm, geom=g1, stiffness_factor=0.5)
ops_assign_floor_masses(n1, g1, m1)
md = modal_x_properties(g1, 3)
rq = rsa_equivalent_floor_forces(g1, m1, md, rp, 1.5)
T1 = max(md, key=lambda d: d["ratio"])["T"]
check("1-storey RSA-equivalent base shear = m·Sd(T1) [kN]", rq["Fb_kN"], m1[1] * design_spectrum_ordinate(T1, rp, 1.5) / 1000, rtol=0.01)

# ------------------------------------------------------------
# 6) Jacket detailing (EC2 8.2(2), EC8 5.4.3.2.2) and axial-force diagram
# ------------------------------------------------------------
from framekit.jacketing import build_jacketed_section, ec2_min_clear_spacing
col.section_role = "column"
js = build_jacketed_section(col, "column", 75.0, stirrup_spacing_mm=100.0, ratio=1.5)
chk = js.jacket_check
check("Jacket 75 mm, +150% steel: clear spacing >= EC2 minimum [mm]",
      float(chk["min_clear_spacing_mm"] >= chk["required_clear_spacing_mm"]), 1.0)
check("Jacket 50 mm is rejected (no room for the cage)",
      float(build_jacketed_section(col, "column", 50.0, ratio=0.5).jacket_check["status"] == "FAIL"), 1.0)
check("EC2 8.2(2) minimum clear spacing, Ø16, dg = 20 mm [mm]", ec2_min_clear_spacing(16, 20), 25.0)

ops_build_model(n1, e1, 30e9, 200e9, col, bm, geom=g1)
swc = compute_self_weight(e1, mat, bm, col)
ops_apply_combo_loads(n1, e1, swc)
ops_run_static()
r = _nvm_arrays_for_element(1, e1, swc, n1, npts=11)
w_c = 2500 * g * (col.b_mm / 1000) * (col.h_mm / 1000) / 1000
check("Column axial force varies with self-weight: N(top) - N(bottom) [kN]", (r[2][-1] - r[2][0]) / 1000, w_c * 3.0)

print("\n" + ("ALL CHECKS PASSED" if all(RESULTS) else f"{RESULTS.count(False)} CHECK(S) FAILED") + f"  ({len(RESULTS)} checks)")
