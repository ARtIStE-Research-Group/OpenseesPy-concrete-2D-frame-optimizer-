"""ARCOPY vs SAP2000 for the two-storey, two-bay portal (examples/test_rc_portal; Section 3 of the paper).

Run the portal example first (python main.py < examples/test_rc_portal/inputs.txt), then this script.

ARCOPY side : the same OpenSees model and the same combination code that ARCOPY uses in the run
              (checkpoints of the portal run in test_rc_portal/_checkpoints).
SAP2000 side: the tables exported by sap_build_test_portal.py (results_static, results_seismic).

Output (in OUT): paired values (pairs_*.csv), statistics (statistics.csv / .md), summary.json.

Sign conventions (plane frame X-Z in SAP2000, x-y in ARCOPY):
  beams   : N = P,  V = -V2,  M =  M3   (ARCOPY: N tension +, V = dM/dx, M sagging +)
  columns : N = P,  V =  V2,  M = -M3   (SAP2000 local 2 = +X for vertical members)
  joints  : ux = U1, uy = U3, rz = -R2  (SAP2000 +Y points into the page)
  supports: Rx = F1, Ry = F3, Mz = -M2
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARC = os.environ.get("ARCOPY_DIR", os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # repository root
RUN = os.path.join(ARC, "test_rc_portal", "_checkpoints")
SAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "test_portal")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
SPEC = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "portal_data.json")   # spectrum given to SAP2000

sys.path.insert(0, os.path.join(ARC, "frame_app"))
import openseespy.opensees as ops                                     # noqa: E402
from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static  # noqa: E402
from framekit.diagrams import _nvm_arrays_for_element                 # noqa: E402
from framekit.superposition import is_group_combo, envelope          # noqa: E402

NST = 11                       # output stations per member in SAP2000 (both ends included)
BASIC = ["SW", "G", "live01", "live02", "live03", "live04", "W+", "W-"]
COMBOS = ["CROWD", "CROWD_SEISM", "WIND", "ULS_Q_leading_W+", "ULS_Q_leading_W-", "ULS_W+_leading",
          "ULS_W-_leading", "SLS_char_W+", "SLS_char_W-", "SLS_quasi_perm"]

st = pickle.load(open(os.path.join(RUN, "step_03_static.pkl"), "rb"))
rs = pickle.load(open(os.path.join(RUN, "step_07_rsa.pkl"), "rb"))
nodes, elements, cases, combos = st["nodes"], st["elements"], st["cases"], st["combos"]
FREE = [n for n in nodes if nodes[n][1] > 0]
BASE = [n for n in nodes if nodes[n][1] == 0]


# ------------------------------------------------------------------ ARCOPY
def build():
    ops_build_model(nodes, elements, st["Ec_Pa"], st["Es_Pa"], st["column_sec"], st["beam_sec"], geom=st["geom"],
                    use_rigid_diaphragm=False, walls=st["walls"], wall_tw_mm=st["wall_tw_mm"],
                    wall_E_Pa=st["wall_E_Pa"])


def analyse(loads):
    """N, V, M [kN, kNm] at the 11 stations of every member, joint displacements, support reactions."""
    build()
    ops_apply_combo_loads(nodes, elements, loads)
    if ops_run_static() != 0:
        raise RuntimeError("static analysis failed")
    ops.reactions()
    f = {}
    for eid in elements:
        _, _, N, V, M, L, _, _ = _nvm_arrays_for_element(eid, elements, loads, nodes, npts=NST)
        f[eid] = np.vstack([N, V, M]) / 1000.0
    u = {n: np.array([ops.nodeDisp(n, 1), ops.nodeDisp(n, 2), ops.nodeDisp(n, 3)]) for n in nodes}
    r = {n: np.array(ops.nodeReaction(n)) / np.array([1000.0, 1000.0, 1000.0]) for n in BASE}
    return {"f": f, "u": u, "r": r}


arc = {c: analyse(cases[c]) for c in BASIC}


def arc_combo(name):
    """(max, min) dictionaries of a combination, with ARCOPY's own combination rules."""
    loads = combos[name]["loads"]
    if not is_group_combo(loads):
        res = analyse(loads)
        return res, res
    hi = {"f": {}, "u": {}, "r": {}}
    lo = {"f": {}, "u": {}, "r": {}}
    for key in ("f", "u", "r"):
        for k in arc["SW"][key]:
            mx, mn = envelope(loads, {c: arc[c][key][k] for c in loads.case_names()})
            hi[key][k], lo[key][k] = mx, mn
    return hi, lo


arc_c = {c: arc_combo(c) for c in COMBOS}

# ------------------------------------------------------------------ SAP2000
ef = pd.read_csv(os.path.join(SAP, "results_static", "Element_Forces___Frames.csv"))
jd = pd.read_csv(os.path.join(SAP, "results_static", "Joint_Displacements.csv"))
jr = pd.read_csv(os.path.join(SAP, "results_static", "Joint_Reactions.csv"))
for d in (ef, jd, jr):
    d["StepType"] = d["StepType"].fillna("")

LEN = {e: float(np.hypot(nodes[elements[e][1]][0] - nodes[elements[e][0]][0],
                         nodes[elements[e][1]][1] - nodes[elements[e][0]][1])) for e in elements}


def sap_forces(case, step=""):
    """{eid: 3 x 11 array of N, V, M in ARCOPY signs}; for a Max/Min row the sign change swaps Max and Min."""
    out = {}
    for eid in elements:
        col = elements[eid][2] == "column"
        sv, sm = (1.0, -1.0) if col else (-1.0, 1.0)

        def rows(stp):
            d = ef[(ef.Frame == eid) & (ef.OutputCase == case) & (ef.StepType == stp)].copy()
            d["i"] = np.rint(d.Station / LEN[eid] * (NST - 1)).astype(int)
            return d.groupby("i").first().reindex(range(NST))

        if step == "":
            d = rows("")
            out[eid] = np.vstack([d.P.values, sv * d.V2.values, sm * d.M3.values])
        else:
            other = "Min" if step == "Max" else "Max"
            d, o = rows(step), rows(other)
            out[eid] = np.vstack([d.P.values, (d if sv > 0 else o).V2.values * sv,
                                  (d if sm > 0 else o).M3.values * sm])
    return out


def sap_joint(case, step=""):
    u, r = {}, {}
    for n in nodes:
        d = jd[(jd.Joint == n) & (jd.OutputCase == case) & (jd.StepType == step)].iloc[0]
        o = jd[(jd.Joint == n) & (jd.OutputCase == case) &
               (jd.StepType == ({"Max": "Min", "Min": "Max"}.get(step, "")))].iloc[0]
        u[n] = np.array([d.U1, d.U3, -o.R2])
    for n in BASE:
        d = jr[(jr.Joint == n) & (jr.OutputCase == case) & (jr.StepType == step)].iloc[0]
        o = jr[(jr.Joint == n) & (jr.OutputCase == case) &
               (jr.StepType == ({"Max": "Min", "Min": "Max"}.get(step, "")))].iloc[0]
        r[n] = np.array([d.F1, d.F3, -o.M2])
    return {"f": sap_forces(case, step), "u": u, "r": r}


def has_steps(case):
    return set(ef[ef.OutputCase == case].StepType.unique()) == {"Max", "Min"}


# ------------------------------------------------------------------ pairs
rows = []
QN = ["N", "V", "M"]
UNIT_F = ["kN", "kN", "kNm"]


def add_pairs(label, kind, a, s):
    for eid in elements:
        for q in range(3):
            for i in range(NST):
                rows.append({"set": kind, "case": label, "quantity": QN[q], "unit": UNIT_F[q], "item": f"E{eid}",
                             "etype": elements[eid][2], "station": i, "x_m": LEN[eid] * i / (NST - 1),
                             "arcopy": a["f"][eid][q, i], "sap2000": s["f"][eid][q, i]})
    for n in FREE:
        for q, (nm, un, sc) in enumerate([("ux", "mm", 1e3), ("uy", "mm", 1e3), ("rz", "mrad", 1e3)]):
            rows.append({"set": kind, "case": label, "quantity": nm, "unit": un, "item": f"J{n}", "etype": "joint",
                         "station": 0, "x_m": 0.0, "arcopy": a["u"][n][q] * sc, "sap2000": s["u"][n][q] * sc})
    for n in BASE:
        for q, (nm, un) in enumerate([("Rx", "kN"), ("Ry", "kN"), ("Mz", "kNm")]):
            rows.append({"set": kind, "case": label, "quantity": nm, "unit": un, "item": f"J{n}", "etype": "support",
                         "station": 0, "x_m": 0.0, "arcopy": a["r"][n][q], "sap2000": s["r"][n][q]})


for c in BASIC:
    add_pairs(c, "basic case", arc[c], sap_joint(c))
for c in COMBOS:
    hi, lo = arc_c[c]
    if has_steps(c):
        add_pairs(c + " max", "combination", hi, sap_joint(c, "Max"))
        add_pairs(c + " min", "combination", lo, sap_joint(c, "Min"))
    else:
        add_pairs(c, "combination", hi, sap_joint(c))

P = pd.DataFrame(rows)
P["diff"] = P.arcopy - P.sap2000
# normalised difference: difference / largest |SAP2000 value| of the same quantity in the same case
P["scale"] = P.groupby(["case", "quantity"]).sap2000.transform(lambda v: np.max(np.abs(v)))
P["e_norm_pct"] = np.where(P.scale > 0, P["diff"] / P.scale * 100.0, 0.0)
# relative difference, only where the value is not small (|SAP2000| >= 5 % of the scale)
big = P.sap2000.abs() >= 0.05 * P.scale
P["e_rel_pct"] = np.where(big, P["diff"] / P.sap2000.where(P.sap2000 != 0, np.nan) * 100.0, np.nan)
# SAP2000 prints forces with 3 decimals (kN), moments with 4 (kNm), displacements and rotations with 6 (m, rad):
# a difference within half of that last digit is the rounding of the SAP2000 table
STEP = {"N": 1e-3, "V": 1e-3, "Rx": 1e-3, "Ry": 1e-3, "M": 1e-4, "Mz": 1e-4, "ux": 1e-3, "uy": 1e-3, "rz": 1e-3}
P["step"] = P.quantity.map(STEP)
P["diff_steps"] = P["diff"] / P.step
P.to_csv(os.path.join(OUT, "pairs_static.csv"), index=False)

# drifts of the wind cases (column lines, storeys)
geom = st["geom"]
drows = []
for c in ("W+", "W-"):
    s = sap_joint(c)
    for stn in (1, 2):
        for ix in range(geom.num_bays + 1):
            top = [n for n in nodes if nodes[n] == (sum(geom.bay_lengths[:ix]), stn * geom.floor_height)][0]
            bot = [n for n in nodes if nodes[n] == (sum(geom.bay_lengths[:ix]), (stn - 1) * geom.floor_height)][0]
            da = (arc[c]["u"][top][0] - arc[c]["u"][bot][0]) / geom.floor_height * 100
            ds = (s["u"][top][0] - s["u"][bot][0]) / geom.floor_height * 100
            drows.append({"case": c, "storey": stn, "line": ix + 1, "arcopy_pct": da, "sap2000_pct": ds})
D = pd.DataFrame(drows)
D["diff_pct_of_value"] = (D.arcopy_pct - D.sap2000_pct) / D.sap2000_pct * 100
D.to_csv(os.path.join(OUT, "pairs_drifts.csv"), index=False)

# ------------------------------------------------------------------ modal and RSA
rr = rs["rsa_results"]
T_a = np.asarray(rr["periods"], float)
mpart = rs["df_modal_part"]
m_arc = float(sum(rs["floor_masses"].values())) / 1000.0               # t
V_arc = float(rr["base_shear"]) / 1000.0


def compare_seismic(folder):
    """Modal and RSA comparison with one SAP2000 seismic model (folder with its exported tables)."""
    mp = pd.read_csv(os.path.join(folder, "Modal_Participating_Mass_Ratios.csv"))
    jm = pd.read_csv(os.path.join(folder, "Assembled_Joint_Masses.csv"))
    jm["Joint"] = pd.to_numeric(jm.Joint, errors="coerce")
    br = pd.read_csv(os.path.join(folder, "Base_Reactions.csv"))
    jds = pd.read_csv(os.path.join(folder, "Joint_Displacements.csv"))
    efs = pd.read_csv(os.path.join(folder, "Element_Forces___Frames.csv"))
    modal = []
    for k in (1, 2):
        sm = mp[mp.StepNum == k].iloc[0]
        sel = lambda n: jds[(jds.Joint == n) & (jds.OutputCase == "MODAL") & (jds.StepNum == k)].iloc[0]  # noqa
        phi_s = np.concatenate([[sel(n).U1, sel(n).U3] for n in FREE])
        phi_a = np.concatenate([[rr["modal_node_disp"][k][n][0], rr["modal_node_disp"][k][n][1]] for n in FREE])
        mac = float(np.dot(phi_a, phi_s) ** 2 / (np.dot(phi_a, phi_a) * np.dot(phi_s, phi_s)))
        modal.append({"mode": k, "T_arcopy_s": T_a[k - 1], "T_sap2000_s": float(sm.Period),
                      "T_diff_pct": (T_a[k - 1] - sm.Period) / sm.Period * 100,
                      "mass_arcopy": float(mpart.Mass_ratio.iloc[k - 1]), "mass_sap2000": float(sm.UX),
                      "MAC": mac,
                      "shape_arcopy": [float(v) for v in phi_a], "shape_sap2000": [float(v) for v in phi_s]})
    M = pd.DataFrame(modal)
    m_sap = float(jm[jm.Joint.isin(FREE)].U1.sum())
    V_sap = float(br[br.OutputCase == "RSA_X"].GlobalFX.iloc[0])
    rows_ = [{"quantity": "base shear", "item": "base", "etype": "base", "unit": "kN", "arcopy": V_arc,
              "sap2000": V_sap}]
    for n in FREE:
        d = jds[(jds.Joint == n) & (jds.OutputCase == "RSA_X")].iloc[0]
        rows_.append({"quantity": "ux", "item": f"J{n}", "etype": "joint", "unit": "mm",
                      "arcopy": rr["node_displacements"][n][0] * 1e3, "sap2000": d.U1 * 1e3})
    for eid in elements:
        a = np.abs(np.asarray(rr["element_forces"][eid], float)) / 1000.0
        d0 = efs[(efs.Frame == eid) & (efs.OutputCase == "RSA_X")].copy()
        d0["i"] = np.rint(d0.Station / LEN[eid] * (NST - 1)).astype(int)
        for end, stn in (("i", 0), ("j", NST - 1)):
            d = d0[d0.i == stn].iloc[0]
            off = 0 if end == "i" else 3
            for q, col, un in ((0, "P", "kN"), (1, "V2", "kN"), (2, "M3", "kNm")):
                if abs(d[col]) < 1e-3 and a[off + q] < 1e-3:
                    continue
                rows_.append({"quantity": QN[q], "item": f"E{eid}{end}", "etype": elements[eid][2], "unit": un,
                              "arcopy": a[off + q], "sap2000": abs(float(d[col]))})
    R = pd.DataFrame(rows_)
    R["diff_pct"] = (R.arcopy - R.sap2000) / R.sap2000 * 100
    return M, R, m_sap, V_sap


SEIS = {}
for tag, sub in (("sap_mass", "results_seismic"), ("same_mass", "results_seismic_same_mass")):
    folder = os.path.join(SAP, sub)
    if os.path.isdir(folder):
        SEIS[tag] = compare_seismic(folder)
        M_, R_, _, _ = SEIS[tag]
        M_.drop(columns=["shape_arcopy", "shape_sap2000"]).to_csv(os.path.join(OUT, f"modal_{tag}.csv"), index=False)
        R_.to_csv(os.path.join(OUT, f"pairs_rsa_{tag}.csv"), index=False)
        json.dump({str(int(r["mode"])): {"arcopy": r.shape_arcopy, "sap2000": r.shape_sap2000}
                   for _, r in M_.iterrows()}, open(os.path.join(OUT, f"shapes_{tag}.json"), "w"))

# expected RSA ratio (first SAP2000 model) from the two known input differences: g and the spectrum interpolation
expl = {}
if SPEC and "sap_mass" in SEIS:
    M1, _, m_sap1, _ = SEIS["sap_mass"]
    sp = json.load(open(SPEC))
    Tt, Ss = np.array(sp["spectrum_T"]), np.array(sp["spectrum_Sd"])
    from framekit.ec8_seismic import design_spectrum_ordinate
    Sa_exact = float(design_spectrum_ordinate(T_a[0], rs["rsa_params"], rs["q_beh"]))
    Sa_sap = float(np.interp(M1.T_sap2000_s.iloc[0], Tt, Ss))
    expl = {"Sa1_exact": Sa_exact, "Sa1_sap_interp": Sa_sap, "Sa_ratio": Sa_sap / Sa_exact,
            "mass_ratio_g": 9.81 / 9.80665, "mass_ratio_table_rounded": m_sap1 / m_arc,
            "T1_ratio_sq": (M1.T_sap2000_s.iloc[0] / T_a[0]) ** 2}
    expl["expected_V_ratio"] = expl["Sa_ratio"] * expl["mass_ratio_g"]
    expl["observed_V_ratio"] = SEIS["sap_mass"][3] / V_arc
    expl["expected_u_ratio"] = expl["Sa_ratio"] * expl["T1_ratio_sq"]


# ------------------------------------------------------------------ statistics
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def stats(df, a="arcopy", s="sap2000", e="e_norm_pct"):
    x, y, ev = df[s].values, df[a].values, df[e].values
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if np.std(x) > 0 else np.nan
    sd = float(np.std(ev, ddof=1)) if len(ev) > 1 else 0.0
    return {"n": int(len(df)), "max_abs_diff": float(np.max(np.abs(y - x))), "bias_pct": float(np.mean(ev)),
            "sd_pct": sd, "mae_pct": float(np.mean(np.abs(ev))), "rmse_pct": float(np.sqrt(np.mean(ev ** 2))),
            "max_abs_pct": float(np.max(np.abs(ev))), "loa_low_pct": float(np.mean(ev) - 1.96 * sd),
            "loa_high_pct": float(np.mean(ev) + 1.96 * sd), "R2": r2, "CCC": ccc(x, y)}


S = []
for kind in ("basic case", "combination"):
    for q in ["N", "V", "M", "ux", "uy", "rz", "Rx", "Ry", "Mz"]:
        d = P[(P.set == kind) & (P.quantity == q)]
        S.append({"group": f"static {kind}", "quantity": q, "unit": d.unit.iloc[0], **stats(d)})
for tag, (M_, R_, _, _) in SEIS.items():
    R_["e"] = R_.diff_pct
    for q, et in (("base shear", "base"), ("ux", "joint"), ("N", "column"), ("N", "beam"), ("V", None), ("M", None)):
        d = R_[(R_.quantity == q) & ((R_.etype == et) if et else True)]
        if len(d):
            S.append({"group": f"RSA ({tag})", "quantity": q + (f" {et}s" if et in ("column", "beam") else ""),
                      "unit": d.unit.iloc[0], **stats(d, e="e")})
S = pd.DataFrame(S)
S.to_csv(os.path.join(OUT, "statistics.csv"), index=False)

summary = {
    "static_pairs": int(len(P)),
    "static_max_abs_diff_force_kN_kNm": float(P[P.quantity.isin(["N", "V", "M", "Rx", "Ry", "Mz"])]["diff"].abs().max()),
    "static_max_abs_e_norm_pct": float(P.e_norm_pct.abs().max()),
    "static_max_abs_e_rel_pct": float(np.nanmax(np.abs(P.e_rel_pct))),
    "static_max_abs_diff_in_rounding_steps": float(P.diff_steps.abs().max()),
    "static_diff_steps_mean_sd": [float(P.diff_steps.mean()), float(P.diff_steps.std())],
    "static_share_exact_to_last_digit": float((P.diff_steps.abs() <= 0.5 + 1e-6).mean()),
    "drift_max_abs_diff_pct_of_value": float(D.diff_pct_of_value.abs().max()),
    "seismic": {tag: {"mass_t": {"arcopy": m_arc, "sap2000": ms, "ratio": ms / m_arc},
                      "modal": M_.drop(columns=["shape_arcopy", "shape_sap2000"]).to_dict("records"),
                      "base_shear_kN": {"arcopy": V_arc, "sap2000": vs, "diff_pct": (V_arc - vs) / vs * 100},
                      "rsa_max_abs_diff_pct": float(R_.diff_pct.abs().max()),
                      "rsa_col_N_max_abs_diff_pct":
                          float(R_[(R_.quantity == "N") & (R_.etype == "column")].diff_pct.abs().max()),
                      "rsa_without_beam_N_max_abs_diff_pct":
                          float(R_[~((R_.quantity == "N") & (R_.etype == "beam"))].diff_pct.abs().max()),
                      "rsa_beam_N": R_[(R_.quantity == "N") & (R_.etype == "beam")][["item", "arcopy", "sap2000"]]
                          .to_dict("records")}
                for tag, (M_, R_, ms, vs) in SEIS.items()},
    "rsa_explanation_sap_mass": expl,
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)

# Member-force table of the paper: largest |N|, |V|, |M| of each member over the four ULS combinations
uls = P[P.case.str.startswith("ULS") & P.quantity.isin(["N", "V", "M"])]
t10 = []
for eid, lab in ((1, "Column 1"), (2, "Column 2"), (4, "Column 4"), (5, "Column 5"), (7, "Beam 7"), (9, "Beam 9")):
    r = {"member": eid, "label_frame": "Portal", "label_member": lab}
    for q in ("N", "V", "M"):
        d = uls[(uls["item"] == f"E{eid}") & (uls.quantity == q)]
        a_, s_ = float(d.arcopy.abs().max()), float(d.sap2000.abs().max())
        r.update({f"{q}_A": a_, f"{q}_S": s_, f"{q}_d": (a_ - s_) / s_ * 100})
    t10.append(r)
json.dump(t10, open(os.path.join(OUT, "member_forces_table.json"), "w"), indent=1)

with open(os.path.join(OUT, "statistics.md"), "w") as fh:
    fh.write("# ARCOPY vs SAP2000 - two-storey portal\n\n")
    fh.write("e = (ARCOPY - SAP2000) / max|SAP2000| of the same quantity in the same case x 100 (static);\n")
    fh.write("e = (ARCOPY - SAP2000) / SAP2000 x 100 (RSA).\n\n")
    fh.write(S.to_markdown(index=False, floatfmt=".4g"))
    for tag, (M_, _, _, _) in SEIS.items():
        fh.write(f"\n\n## Modal ({tag})\n\n" +
                 M_.drop(columns=["shape_arcopy", "shape_sap2000"]).to_markdown(index=False, floatfmt=".6g"))
    fh.write("\n\n## Drifts under W+ and W- [% of h]\n\n" + D.to_markdown(index=False, floatfmt=".5g"))
    fh.write("\n\n## Summary\n\n```\n" + json.dumps(summary, indent=2, default=float) + "\n```\n")
print(S.to_string())
print(json.dumps(summary, indent=2, default=float))
