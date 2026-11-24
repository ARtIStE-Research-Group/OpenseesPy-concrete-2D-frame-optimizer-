"""
Builds the six frames of the ARCOPY paper in SAP2000 (existing frames, before retrofit) and exports
the results that are compared with ARCOPY.

All data come from `six_frames_data.json` (same folder), which was written from the ARCOPY runs:
nodes, members, sections, transformed-section stiffness (property modifiers on A and I33), loads,
combinations, floor masses, number of modes and design spectrum. Nothing is typed by hand.

For each frame two models are built and run:
  <frame>_static.sdb   I33 x 1.0 : SW, G, live01, W+, W- and the nine ULS/SLS combinations
  <frame>_seismic.sdb  I33 x 0.5 : modal analysis and response spectrum RSA_X (CQC, 5 %)
                       mass as in ARCOPY: the mass of each floor in X only, split equally among the
                       joints of the floor (the sections get a mass modifier of 0, because SAP2000
                       takes the joint masses only together with the element self mass)
Both models: plane frame X-Z, fixed bases, no shear deformation, no rigid end zones,
11 output stations per member.

How to run (Spyder): open this file and press Run (F5). SAP2000 opens, builds and runs the twelve
models and closes. It takes a few minutes.
Output: six_frames/<frame>/results_static/*.csv, six_frames/<frame>/results_seismic/*.csv,
        six_frames/build_log.txt
"""
import csv
import json
import os
import sys
import time

try:
    import comtypes.client
except ImportError:
    sys.exit("The package comtypes is missing: run  pip install comtypes  in the Python environment")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "six_frames")
DATA = json.load(open(os.path.join(HERE, "six_frames_data.json"), encoding="utf-8"))
FRAMES = list(DATA)                     # to run only some frames: FRAMES = ["F4-5B"]
MAT = "CONC_ARCOPY"
LOG = []


def call(label, fn, *args):
    """Calls an OAPI function, logs its return code and returns the full result."""
    try:
        res = fn(*args)
    except Exception as exc:                                   # noqa: BLE001
        LOG.append(f"ERROR {label}: {exc}")
        print(f"  ERROR {label}: {exc}")
        return None
    rc = res[-1] if isinstance(res, (list, tuple)) else res
    LOG.append(f"{'ok   ' if rc == 0 else 'FAIL '} {label} -> {rc}")
    if rc != 0:
        print(f"  FAIL {label} (return code {rc})")
    return res


def start_sap():
    helper = comtypes.client.CreateObject("SAP2000v1.Helper")
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    sap = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
    sap.ApplicationStart()
    return sap


def build(m, d, seismic):
    crack = d["crack"] if seismic else 1.0
    call("InitializeNewModel", m.InitializeNewModel, 6)                 # kN, m, C
    call("NewBlank", m.File.NewBlank)
    call("SetMaterial", m.PropMaterial.SetMaterial, MAT, 2)
    call("SetMPIsotropic", m.PropMaterial.SetMPIsotropic, MAT, d["E_kPa"], 0.2, 1.0e-5)
    call("SetWeightAndMass", m.PropMaterial.SetWeightAndMass, MAT, 1, d["unit_weight"])
    # sections and property modifiers [A, AS2, AS3, J, I22, I33, mass, weight]
    prop = {}
    for kind, s in d["sections"].items():
        prop[kind] = f"{kind.upper()}_{round(s['b'] * 1000)}x{round(s['h'] * 1000)}"
        call(f"SetRectangle {prop[kind]}", m.PropFrame.SetRectangle, prop[kind], MAT, s["h"], s["b"])
        call(f"SetModifiers {prop[kind]}", m.PropFrame.SetModifiers, prop[kind],
             [s["A_mod"], 0.0, 0.0, 1.0, 1.0, s["I33_mod"] * crack, 0.0 if seismic else 1.0, 1.0])
    # joints (ARCOPY numbering) and fixed bases
    for nid, (x, z) in d["nodes"].items():
        call(f"AddCartesian {nid}", m.PointObj.AddCartesian, float(x), 0.0, float(z), "", str(nid))
        if abs(z) < 1e-9:
            call(f"SetRestraint {nid}", m.PointObj.SetRestraint, str(nid), [True] * 6)
    # members (ARCOPY numbering), 11 output stations each
    for eid, (ni, nj, kind) in d["elements"].items():
        call(f"AddByPoint {eid}", m.FrameObj.AddByPoint, str(ni), str(nj), "", prop[kind], str(eid))
        call(f"SetOutputStations {eid}", m.FrameObj.SetOutputStations, str(eid), 2, 0.5, 11, False, False)
    call("SetActiveDOF", m.Analyze.SetActiveDOF, [True, False, True, False, True, False])     # plane frame XZ
    # load patterns: the default DEAD pattern becomes SW (self-weight multiplier 1)
    r1 = call("ChangeName DEAD->SW (pattern)", m.LoadPatterns.ChangeName, "DEAD", "SW")
    r2 = call("ChangeName DEAD->SW (case)", m.LoadCases.ChangeName, "DEAD", "SW")
    if r1 is None or r1 != 0 or r2 is None or r2 != 0:
        call("SetSelfWTMultiplier DEAD 0", m.LoadPatterns.SetSelfWTMultiplier, "DEAD", 0.0)
        call("Add SW", m.LoadPatterns.Add, "SW", 1, 1.0, True)
    for case in d["loads"]:
        ptype = 2 if case == "G" else (8 if case.startswith("W") else 3)   # super dead / other / live
        call(f"Add {case}", m.LoadPatterns.Add, case, ptype, 0.0, True)
    # distributed loads (SW comes from the self weight); gravity dir 10, global X dir 4
    for case, lst in d["loads"].items():
        for eid, direction, val in lst:
            if direction == "vertical":
                call(f"Load {case} e{eid}", m.FrameObj.SetLoadDistributed, str(eid), case, 1, 10, 0.0, 1.0,
                     -val, -val, "Global", True, True)
            else:
                call(f"Load {case} e{eid}", m.FrameObj.SetLoadDistributed, str(eid), case, 1, 4, 0.0, 1.0,
                     val, val, "Global", True, True)
    # combinations: Linear Add, same names and factors as ARCOPY
    for name, items in d["combos"].items():
        call(f"Combo {name}", m.RespCombo.Add, name, 0)
        for cname, sf in items:
            call(f"Combo {name} + {cname}", m.RespCombo.SetCaseList, name, 0, cname, sf)
    if not seismic:
        return
    # mass of ARCOPY: joint masses in X, equal at the joints of each floor; no element or load mass
    call("SetMassSource", m.SourceMass.SetMassSource, "ARCOPY_MASS", True, True, False, True, 1, ["G"], [0.0])
    for k, mass in d["floor_mass_t"].items():
        z = int(k) * d["storey_height"]
        joints = [nid for nid, (x, zz) in d["nodes"].items() if abs(zz - z) < 1e-6]
        for nid in joints:
            call(f"SetMass {nid}", m.PointObj.SetMass, str(nid), [mass / len(joints), 0.0, 0.0, 0.0, 0.0, 0.0],
                 0, True, True)
    # modal and response-spectrum cases
    call("SetNumberModes", m.LoadCases.ModalEigen.SetNumberModes, "MODAL", d["n_modes"], 1)
    T, Sd = d["spectrum_T"], d["spectrum_Sd"]
    call("FuncRS.SetUser", m.Func.FuncRS.SetUser, "ARCOPY_SD", len(T), T, Sd, 0.05)
    rs = m.LoadCases.ResponseSpectrum
    call("RS SetCase", rs.SetCase, "RSA_X")
    call("RS SetModalCase", rs.SetModalCase, "RSA_X", "MODAL")
    if call("RS SetModalComb_1", rs.SetModalComb_1, "RSA_X", 1, 1.0, 0.0, 1, 0.0) is None:
        call("RS SetModalComb", rs.SetModalComb, "RSA_X", 1, 1.0, 0.0, 0.0)
    call("RS SetDirComb", rs.SetDirComb, "RSA_X", 1, 0.0)
    call("RS SetLoads", rs.SetLoads, "RSA_X", 1, ["U1"], ["ARCOPY_SD"], [1.0], ["Global"], [0.0])
    call("RS SetDampConstant", rs.SetDampConstant, "RSA_X", 0.05)


def export(m, outdir, tables, cases, combos):
    os.makedirs(outdir, exist_ok=True)
    call("SetLoadCasesSelectedForDisplay", m.DatabaseTables.SetLoadCasesSelectedForDisplay, cases)
    call("SetLoadCombinationsSelectedForDisplay", m.DatabaseTables.SetLoadCombinationsSelectedForDisplay, combos)
    for key in tables:
        res = call(f"Table {key}", m.DatabaseTables.GetTableForDisplayArray, key, [], "", 0, [], 0, [])
        if res is None or res[-1] != 0:
            continue
        fields, nrec, data = list(res[-4]), int(res[-3]), list(res[-2])
        n = len(fields)
        fname = "".join(c if c.isalnum() else "_" for c in key) + ".csv"
        with open(os.path.join(outdir, fname), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(fields)
            for i in range(nrec):
                w.writerow(data[i * n:(i + 1) * n])
        LOG.append(f"      {key}: {nrec} rows")


def run_model(m, d, name, seismic):
    tag = "seismic" if seismic else "static"
    LOG.append(f"===== {name}_{tag} =====")
    print(f"Building {name}_{tag} ...")
    build(m, d, seismic)
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    call("Save", m.File.Save, os.path.join(folder, f"{name}_{tag}.sdb"))
    call("SetRunCaseFlag all", m.Analyze.SetRunCaseFlag, "", True, True)
    print("  running ...")
    call("RunAnalysis", m.Analyze.RunAnalysis)
    m.SetPresentUnits(6)
    if seismic:
        export(m, os.path.join(folder, "results_seismic"),
               ["Modal Periods And Frequencies", "Modal Participating Mass Ratios", "Joint Displacements",
                "Base Reactions", "Element Forces - Frames", "Assembled Joint Masses"], ["MODAL", "RSA_X"], [])
        export(m, os.path.join(folder, "model_check"),
               ["Mass Source", "Frame Section Properties 01 - General", "Combination Definitions"], [], [])
    else:
        export(m, os.path.join(folder, "results_static"),
               ["Joint Displacements", "Joint Reactions", "Element Forces - Frames", "Base Reactions"],
               ["SW"] + list(d["loads"]), list(d["combos"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    sap = start_sap()
    m = sap.SapModel
    t0 = time.time()
    try:
        for name in FRAMES:
            run_model(m, DATA[name], name, False)
            run_model(m, DATA[name], name, True)
    finally:
        with open(os.path.join(OUT, "build_log.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(LOG))
        sap.ApplicationExit(False)
    n_fail = sum(1 for line in LOG if line.startswith(("FAIL", "ERROR")))
    print(f"\nDone in {time.time() - t0:.0f} s, {n_fail} call(s) failed (see build_log.txt). Results in: {OUT}")


if __name__ == "__main__":
    main()
