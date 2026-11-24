"""ARCOPY vs SAP2000 for the six frames (existing frames, before retrofit; Section 4.1 of the paper).

Run the six baseline runs first (python examples/six_frames/run_all.py F4-5B_base F6-5B_base ...), then this script.

ARCOPY side : the OpenSees model and the combinations of the baseline runs (<frame>_base/_checkpoints).
SAP2000 side: tables exported by sap_build_six_frames.py (six_frames/<frame>/results_static, results_seismic).

Output in OUT: pairs_static.csv, pairs_rsa.csv, modal.csv, drifts.csv, statistics.csv, summary.json

Signs (plane frame X-Z in SAP2000, x-y in ARCOPY):
  beams: N = P, V = -V2, M = M3;  columns: N = P, V = V2, M = -M3;  joints: ux = U1, uy = U3, rz = -R2
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARC = os.environ.get("ARCOPY_DIR", os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # repository root
SAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "six_frames")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ARC, "frame_app"))
import openseespy.opensees as ops                                     # noqa: E402
from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static  # noqa: E402
from framekit.diagrams import _nvm_arrays_for_element                 # noqa: E402

FRAMES = ["F4-5B", "F6-5B", "F8-5B", "F4-7B", "F6-7B", "F8-7B"]
NST = 11
QN = ["N", "V", "M"]
STEP = {"N": 1e-3, "V": 1e-3, "Rx": 1e-3, "Ry": 1e-3, "M": 1e-4, "Mz": 1e-4, "ux": 1e-3, "uy": 1e-3, "rz": 1e-3}


def one_frame(name):
    ck = os.path.join(ARC, name + "_base", "_checkpoints")
    st = pickle.load(open(os.path.join(ck, "step_03_static.pkl"), "rb"))
    rs = pickle.load(open(os.path.join(ck, "step_07_rsa.pkl"), "rb"))
    nodes, elements, cases, combos, geom = st["nodes"], st["elements"], st["cases"], st["combos"], st["geom"]
    groups = st["load_groups"]["groups"]
    FREE = [n for n in nodes if nodes[n][1] > 0]
    BASE = [n for n in nodes if nodes[n][1] == 0]
    LEN = {e: float(np.hypot(nodes[elements[e][1]][0] - nodes[elements[e][0]][0],
                             nodes[elements[e][1]][1] - nodes[elements[e][0]][1])) for e in elements}

    def analyse(loads):
        ops_build_model(nodes, elements, st["Ec_Pa"], st["Es_Pa"], st["column_sec"], st["beam_sec"], geom=geom,
                        use_rigid_diaphragm=False, walls=st["walls"], wall_tw_mm=st["wall_tw_mm"],
                        wall_E_Pa=st["wall_E_Pa"])
        ops_apply_combo_loads(nodes, elements, loads)
        assert ops_run_static() == 0
        ops.reactions()
        f = {}
        for eid in elements:
            _, _, N, V, M, L, _, _ = _nvm_arrays_for_element(eid, elements, loads, nodes, npts=NST)
            f[eid] = np.vstack([N, V, M]) / 1000.0
        u = {n: np.array([ops.nodeDisp(n, 1), ops.nodeDisp(n, 2), ops.nodeDisp(n, 3)]) for n in nodes}
        r = {n: np.array(ops.nodeReaction(n)) / 1000.0 for n in BASE}
        return {"f": f, "u": u, "r": r}

    basic = [c for c in cases if c != "Q"]
    cnames = [c for c in combos if c not in groups]
    arc = {c: analyse(cases[c]) for c in basic}
    arc.update({c: analyse(combos[c]["loads"]) for c in cnames})

    sd = os.path.join(SAP, name, "results_static")
    ef = pd.read_csv(os.path.join(sd, "Element_Forces___Frames.csv"))
    jd = pd.read_csv(os.path.join(sd, "Joint_Displacements.csv"))
    jr = pd.read_csv(os.path.join(sd, "Joint_Reactions.csv"))
    ef["i"] = np.rint(ef.Station / ef.Frame.map(LEN) * (NST - 1)).astype(int)
    rows = []
    for c in basic + cnames:
        kind = "load case" if c in basic else "combination"
        e_c = ef[ef.OutputCase == c].set_index(["Frame", "i"])
        j_c = jd[jd.OutputCase == c].set_index("Joint")
        r_c = jr[jr.OutputCase == c].set_index("Joint")
        for eid in elements:
            col = elements[eid][2] == "column"
            sv, sm = (1.0, -1.0) if col else (-1.0, 1.0)
            d = e_c.loc[eid].reindex(range(NST))
            s = np.vstack([d.P.values, sv * d.V2.values, sm * d.M3.values])
            for q in range(3):
                for i in range(NST):
                    rows.append((name, kind, c, QN[q], f"E{eid}", elements[eid][2], i, arc[c]["f"][eid][q, i], s[q, i]))
        for n in FREE:
            d = j_c.loc[n]
            for q, (nm, v) in enumerate((("ux", d.U1), ("uy", d.U3), ("rz", -d.R2))):
                rows.append((name, kind, c, nm, f"J{n}", "joint", 0, arc[c]["u"][n][q] * 1e3, v * 1e3))
        for n in BASE:
            d = r_c.loc[n]
            for q, (nm, v) in enumerate((("Rx", d.F1), ("Ry", d.F3), ("Mz", -d.M2))):
                rows.append((name, kind, c, nm, f"J{n}", "support", 0, arc[c]["r"][n][q], v))
    P = pd.DataFrame(rows, columns=["frame", "set", "case", "quantity", "item", "etype", "station", "arcopy", "sap2000"])
    P["diff"] = P.arcopy - P.sap2000
    P["diff_steps"] = P["diff"] / P.quantity.map(STEP)

    # roof displacement and largest storey drift under the wind-leading ULS combinations
    h = geom.floor_height
    xs = sorted({nodes[n][0] for n in nodes})
    node_at = {(nodes[n][0], round(nodes[n][1] / h)): n for n in nodes}
    drows = []
    for c in ("ULS_W+_leading", "ULS_W-_leading"):
        j_c = jd[jd.OutputCase == c].set_index("Joint")
        for stn in range(1, geom.num_floors + 1):
            da = max(abs(arc[c]["u"][node_at[(x, stn)]][0] - arc[c]["u"][node_at[(x, stn - 1)]][0]) for x in xs) / h
            ds = max(abs(j_c.loc[node_at[(x, stn)]].U1 - j_c.loc[node_at[(x, stn - 1)]].U1) for x in xs) / h
            drows.append((name, c, stn, da * 100, ds * 100))
    D = pd.DataFrame(drows, columns=["frame", "case", "storey", "arcopy_pct", "sap2000_pct"])

    # modal and RSA
    zd = os.path.join(SAP, name, "results_seismic")
    mp = pd.read_csv(os.path.join(zd, "Modal_Participating_Mass_Ratios.csv"))
    br = pd.read_csv(os.path.join(zd, "Base_Reactions.csv"))
    jds = pd.read_csv(os.path.join(zd, "Joint_Displacements.csv"))
    efs = pd.read_csv(os.path.join(zd, "Element_Forces___Frames.csv"))
    rr = rs["rsa_results"]
    T_a = np.asarray(rr["periods"], float)
    mpart = rs["df_modal_part"]
    mrows = []
    jm = jds[jds.OutputCase == "MODAL"].set_index(["StepNum", "Joint"])
    for k in range(1, len(T_a) + 1):
        sm_ = mp[mp.StepNum == k].iloc[0]
        phi_s = np.concatenate([[jm.loc[(k, n)].U1, jm.loc[(k, n)].U3] for n in FREE])
        phi_a = np.concatenate([[rr["modal_node_disp"][k][n][0], rr["modal_node_disp"][k][n][1]] for n in FREE])
        mac = float(np.dot(phi_a, phi_s) ** 2 / (np.dot(phi_a, phi_a) * np.dot(phi_s, phi_s)))
        mrows.append((name, k, T_a[k - 1], float(sm_.Period), float(mpart.Mass_ratio.iloc[k - 1]), float(sm_.UX), mac))
    M = pd.DataFrame(mrows, columns=["frame", "mode", "T_arcopy", "T_sap2000", "mass_arcopy", "mass_sap2000", "MAC"])
    V_sap = float(br[br.OutputCase == "RSA_X"].GlobalFX.iloc[0])
    V_arc = float(rr["base_shear"]) / 1000.0
    rrows = [(name, "base shear", "base", "base", V_arc, V_sap)]
    jr_ = jds[jds.OutputCase == "RSA_X"].set_index("Joint")
    for n in FREE:
        rrows.append((name, "ux", f"J{n}", "joint", rr["node_displacements"][n][0] * 1e3, jr_.loc[n].U1 * 1e3))
    e_r = efs[efs.OutputCase == "RSA_X"].copy()
    e_r["i"] = np.rint(e_r.Station / e_r.Frame.map(LEN) * (NST - 1)).astype(int)
    e_r = e_r.set_index(["Frame", "i"])
    for eid in elements:
        a = np.abs(np.asarray(rr["element_forces"][eid], float)) / 1000.0
        for end, stn, off in (("i", 0, 0), ("j", NST - 1, 3)):
            d = e_r.loc[(eid, stn)]
            for q, colname in enumerate(("P", "V2", "M3")):
                if abs(d[colname]) < 1e-3 and a[off + q] < 1e-3:
                    continue
                rrows.append((name, QN[q], f"E{eid}{end}", elements[eid][2], a[off + q], abs(float(d[colname]))))
    R = pd.DataFrame(rrows, columns=["frame", "quantity", "item", "etype", "arcopy", "sap2000"])
    R["diff_pct"] = (R.arcopy - R.sap2000) / R.sap2000 * 100
    roof = max(nodes[n][1] for n in nodes)
    top = [n for n in nodes if nodes[n][1] == roof]
    scal = {"frame": name, "members": len(elements), "joints": len(nodes),
            "T1_arcopy": T_a[0], "T1_sap2000": float(mp[mp.StepNum == 1].Period.iloc[0]),
            "Vb_arcopy": V_arc, "Vb_sap2000": V_sap,
            "roof_rsa_arcopy_mm": max(rr["node_displacements"][n][0] for n in top) * 1e3,
            "roof_rsa_sap2000_mm": max(jr_.loc[n].U1 for n in top) * 1e3}
    for c in ("ULS_W+_leading",):
        j_c = jd[jd.OutputCase == c].set_index("Joint")
        scal["roof_wind_arcopy_mm"] = max(abs(arc[c]["u"][n][0]) for n in top) * 1e3
        scal["roof_wind_sap2000_mm"] = max(abs(j_c.loc[n].U1) for n in top) * 1e3
    dd = D[D.case == "ULS_W+_leading"]
    scal["drift_wind_arcopy_pct"] = dd.arcopy_pct.max()
    scal["drift_wind_sap2000_pct"] = dd.sap2000_pct.max()
    return P, D, M, R, scal


def main():
    Ps, Ds, Ms, Rs, Ss = [], [], [], [], []
    for name in FRAMES:
        if not os.path.isdir(os.path.join(SAP, name, "results_static")):
            print("no SAP2000 results for", name)
            continue
        P, D, M, R, s = one_frame(name)
        Ps.append(P); Ds.append(D); Ms.append(M); Rs.append(R); Ss.append(s)
        print(name, len(P), "static values, max |diff| in last digit:", round(P.diff_steps.abs().max(), 3),
              "| T diff max %:", round(((M.T_arcopy - M.T_sap2000) / M.T_sap2000 * 100).abs().max(), 5),
              "| MAC min:", round(M.MAC.min(), 8), "| RSA max %:", round(R.diff_pct.abs().max(), 4), flush=True)
    P, D, M, R = (pd.concat(x, ignore_index=True) for x in (Ps, Ds, Ms, Rs))
    P.to_csv(os.path.join(OUT, "pairs_static.csv"), index=False)
    D.to_csv(os.path.join(OUT, "drifts.csv"), index=False)
    M.to_csv(os.path.join(OUT, "modal.csv"), index=False)
    R.to_csv(os.path.join(OUT, "pairs_rsa.csv"), index=False)
    S = pd.DataFrame(Ss)
    S.to_csv(os.path.join(OUT, "scalars.csv"), index=False)
    st = []
    for name, g in P.groupby("frame", sort=False):
        e = g.diff_steps
        r = R[R.frame == name]
        m = M[M.frame == name]
        st.append({"frame": name, "n_static": len(g), "static_bias_digit": e.mean(), "static_sd_digit": e.std(),
                   "static_max_digit": e.abs().max(), "share_within_half_digit": float((e.abs() <= 0.5 + 1e-6).mean()),
                   "static_max_abs_diff": g["diff"].abs().max(),
                   "T_max_diff_pct": ((m.T_arcopy - m.T_sap2000) / m.T_sap2000 * 100).abs().max(),
                   "MAC_min": m.MAC.min(), "n_rsa": len(r), "rsa_bias_pct": r.diff_pct.mean(),
                   "rsa_sd_pct": r.diff_pct.std(), "rsa_max_pct": r.diff_pct.abs().max()})
    ST = pd.DataFrame(st)
    ST.to_csv(os.path.join(OUT, "statistics.csv"), index=False)
    summary = {"n_static": int(len(P)), "static_max_digit": float(P.diff_steps.abs().max()),
               "static_bias_digit": float(P.diff_steps.mean()), "static_sd_digit": float(P.diff_steps.std()),
               "share_within_half_digit": float((P.diff_steps.abs() <= 0.5 + 1e-6).mean()),
               "n_rsa": int(len(R)), "rsa_max_pct": float(R.diff_pct.abs().max()),
               "rsa_bias_pct": float(R.diff_pct.mean()), "rsa_sd_pct": float(R.diff_pct.std()),
               "MAC_min": float(M.MAC.min()),
               "T_max_diff_pct": float(((M.T_arcopy - M.T_sap2000) / M.T_sap2000 * 100).abs().max())}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(ST.to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
