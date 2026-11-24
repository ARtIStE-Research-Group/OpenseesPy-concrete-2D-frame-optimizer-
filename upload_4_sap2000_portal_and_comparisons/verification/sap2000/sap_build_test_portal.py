"""
Builds the 2-storey, 2-bay test portal of the ARCOPY paper in SAP2000, runs it and exports the
results (SAP2000 OAPI, Python + comtypes). Nothing is typed by hand: every number below comes
from the ARCOPY model of the same frame (examples/test_rc_portal).

How to run (Spyder, on the PC with SAP2000):  open this file and press Run (F5).
SAP2000 opens by itself, builds two models, runs them, exports the results and closes again.

Two models, as in ARCOPY:
  TestPortal_static.sdb    transformed sections (property modifiers on A and I33), used for the
                           static load cases and combinations
  TestPortal_seismic.sdb   the same with I33 x 0.5 (EN 1998-1 4.3.1(7)), used for MODAL and RSA
Both: plane frame XZ (UX, UZ, RY), no shear deformation (shear-area modifiers 0), no end offsets,
E = 30 GPa, unit weight 24.525 kN/m3 (= 2500 kg/m3 x 9.81), mass from elements + G + 0.3 x live,
EC8 design spectrum (q = 1.5, ag = 0.25 g, soil B, type 1, 5 %) as a user function in m/s2.

Output (folder "test_portal" next to this script): the two .sdb models, results_static/*.csv,
results_seismic/*.csv and build_log.txt (every OAPI call and its return code).
Needs: pip install comtypes.  "Invalid class string": run RegisterSAP2000.exe once as administrator.
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_portal")
LOG = []

try:
    import comtypes.client
except ImportError:
    sys.exit("The package comtypes is missing: open the Anaconda Prompt and run  pip install comtypes")

DATA = {
 "nodes": {
  "1": [
   0.0,
   0.0
  ],
  "2": [
   5.0,
   0.0
  ],
  "3": [
   10.0,
   0.0
  ],
  "4": [
   0.0,
   4.0
  ],
  "5": [
   5.0,
   4.0
  ],
  "6": [
   10.0,
   4.0
  ],
  "7": [
   0.0,
   8.0
  ],
  "8": [
   5.0,
   8.0
  ],
  "9": [
   10.0,
   8.0
  ]
 },
 "elements": {
  "1": [
   1,
   4,
   "column"
  ],
  "2": [
   2,
   5,
   "column"
  ],
  "3": [
   3,
   6,
   "column"
  ],
  "4": [
   4,
   7,
   "column"
  ],
  "5": [
   5,
   8,
   "column"
  ],
  "6": [
   6,
   9,
   "column"
  ],
  "7": [
   4,
   5,
   "beam"
  ],
  "8": [
   5,
   6,
   "beam"
  ],
  "9": [
   7,
   8,
   "beam"
  ],
  "10": [
   8,
   9,
   "beam"
  ]
 },
 "loads": {
  "G": [
   [
    7,
    "vertical",
    -24.0
   ],
   [
    8,
    "vertical",
    -24.0
   ],
   [
    9,
    "vertical",
    -20.0
   ],
   [
    10,
    "vertical",
    -20.0
   ]
  ],
  "W+": [
   [
    1,
    "horizontal",
    1.4
   ],
   [
    4,
    "horizontal",
    2.8
   ]
  ],
  "W-": [
   [
    3,
    "horizontal",
    -1.4
   ],
   [
    6,
    "horizontal",
    -2.8
   ]
  ],
  "live01": [
   [
    7,
    "vertical",
    -8.0
   ]
  ],
  "live02": [
   [
    8,
    "vertical",
    -8.0
   ]
  ],
  "live03": [
   [
    9,
    "vertical",
    -6.0
   ]
  ],
  "live04": [
   [
    10,
    "vertical",
    -6.0
   ]
  ],
  "SW": [
   [
    1,
    "vertical",
    -2.20725
   ],
   [
    2,
    "vertical",
    -2.20725
   ],
   [
    3,
    "vertical",
    -2.20725
   ],
   [
    4,
    "vertical",
    -2.20725
   ],
   [
    5,
    "vertical",
    -2.20725
   ],
   [
    6,
    "vertical",
    -2.20725
   ],
   [
    7,
    "vertical",
    -3.67875
   ],
   [
    8,
    "vertical",
    -3.67875
   ],
   [
    9,
    "vertical",
    -3.67875
   ],
   [
    10,
    "vertical",
    -3.67875
   ]
  ]
 },
 "spectrum_T": [
  0.0,
  0.02,
  0.03,
  0.04,
  0.05,
  0.06,
  0.07,
  0.08,
  0.09,
  0.1,
  0.11,
  0.12,
  0.13,
  0.14,
  0.15,
  0.16,
  0.18,
  0.2,
  0.22,
  0.24,
  0.26,
  0.28,
  0.3,
  0.32,
  0.34,
  0.36,
  0.38,
  0.4,
  0.42,
  0.44,
  0.46,
  0.48,
  0.5,
  0.55,
  0.6,
  0.65,
  0.7,
  0.75,
  0.8,
  0.85,
  0.9,
  0.95,
  1.0,
  1.05,
  1.1,
  1.15,
  1.2,
  1.25,
  1.3,
  1.35,
  1.4,
  1.45,
  1.5,
  1.55,
  1.6,
  1.65,
  1.7,
  1.75,
  1.8,
  1.85,
  1.9,
  1.95,
  2.0,
  2.1,
  2.2,
  2.3,
  2.4,
  2.5,
  2.6,
  2.7,
  2.8,
  2.9,
  3.0,
  3.1,
  3.2,
  3.3,
  3.4,
  3.5,
  3.6,
  3.7,
  3.8,
  3.9,
  4.0
 ],
 "spectrum_Sd": [
  1.962,
  2.3544,
  2.5506,
  2.7468,
  2.943,
  3.1392,
  3.3354,
  3.5316,
  3.7278,
  3.9240000000000004,
  4.1202,
  4.316400000000001,
  4.5126,
  4.7088,
  4.904999999999999,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.905,
  4.459090909090909,
  4.0875,
  3.773076923076923,
  3.503571428571429,
  3.27,
  3.065625,
  2.8852941176470592,
  2.725,
  2.5815789473684214,
  2.4525,
  2.335714285714286,
  2.2295454545454545,
  2.1326086956521744,
  2.04375,
  1.9620000000000002,
  1.8865384615384615,
  1.8166666666666667,
  1.7517857142857145,
  1.6913793103448278,
  1.635,
  1.582258064516129,
  1.5328125,
  1.4863636363636366,
  1.4426470588235296,
  1.4014285714285715,
  1.3625,
  1.3256756756756756,
  1.2907894736842107,
  1.2576923076923079,
  1.22625,
  1.1122448979591837,
  1.0134297520661155,
  0.9272211720226845,
  0.8515625000000001,
  0.7848,
  0.7255917159763313,
  0.6728395061728395,
  0.6256377551020409,
  0.5832342449464923,
  0.545,
  0.5104058272632674,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005,
  0.49050000000000005
 ],
 "q": 1.5,
 "modifiers": {
  "beam": {
   "A": 1.1117010721276372,
   "I33": 1.2144660584850633
  },
  "column": {
   "A": 1.1396263401595463,
   "I33": 1.2680825731063288
  }
 },
 "crack": 0.5,
 "E_kPa": 30000000.0
}

MAT = "C25_30_ARCOPY"
PROPS = {"beam": ("BEAM_500x300", 0.30, 0.50), "column": ("COL_300x300", 0.30, 0.30)}   # name, T3 (depth), T2
LIVE = ["live01", "live02", "live03", "live04"]
COMBOS = {   # name: (type, [(name, is_combo, factor)]); type 0 linear, 1 envelope, 4 range add
    "CROWD": (4, [(c, False, 1.0) for c in LIVE]),
    "CROWD_SEISM": (0, [(c, False, 1.0) for c in LIVE]),
    "WIND": (1, [("W+", False, 1.0), ("W-", False, 1.0)]),
    "ULS_Q_leading_W+": (0, [("SW", False, 1.35), ("G", False, 1.35), ("CROWD", True, 1.5), ("W+", False, 0.9)]),
    "ULS_Q_leading_W-": (0, [("SW", False, 1.35), ("G", False, 1.35), ("CROWD", True, 1.5), ("W-", False, 0.9)]),
    "ULS_W+_leading": (0, [("SW", False, 1.35), ("G", False, 1.35), ("CROWD", True, 1.05), ("W+", False, 1.5)]),
    "ULS_W-_leading": (0, [("SW", False, 1.35), ("G", False, 1.35), ("CROWD", True, 1.05), ("W-", False, 1.5)]),
    "SLS_char_W+": (0, [("SW", False, 1.0), ("G", False, 1.0), ("CROWD", True, 1.0), ("W+", False, 0.6)]),
    "SLS_char_W-": (0, [("SW", False, 1.0), ("G", False, 1.0), ("CROWD", True, 1.0), ("W-", False, 0.6)]),
    "SLS_quasi_perm": (0, [("SW", False, 1.0), ("G", False, 1.0)] + [(c, False, 0.3) for c in LIVE]),
}
STATIC_CASES = ["SW", "G"] + LIVE + ["W+", "W-"]


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


def build(m, crack):
    call("InitializeNewModel", m.InitializeNewModel, 6)                 # kN, m, C
    call("NewBlank", m.File.NewBlank)
    # material: E = 30 GPa, nu = 0.2, unit weight 24.525 kN/m3
    call("SetMaterial", m.PropMaterial.SetMaterial, MAT, 2)
    call("SetMPIsotropic", m.PropMaterial.SetMPIsotropic, MAT, DATA["E_kPa"], 0.2, 1.0e-5)
    call("SetWeightAndMass", m.PropMaterial.SetWeightAndMass, MAT, 1, 24.525)
    # sections and property modifiers [A, AS2, AS3, J, I22, I33, mass, weight]
    for kind, (name, t3, t2) in PROPS.items():
        call(f"SetRectangle {name}", m.PropFrame.SetRectangle, name, MAT, t3, t2)
        mod = DATA["modifiers"][kind]
        call(f"SetModifiers {name}", m.PropFrame.SetModifiers, name,
             [mod["A"], 0.0, 0.0, 1.0, 1.0, mod["I33"] * crack, 1.0, 1.0])
    # joints (ARCOPY numbering) and fixed bases
    for nid, (x, z) in DATA["nodes"].items():
        call(f"AddCartesian {nid}", m.PointObj.AddCartesian, float(x), 0.0, float(z), "", str(nid))
        if abs(z) < 1e-9:
            call(f"SetRestraint {nid}", m.PointObj.SetRestraint, str(nid), [True] * 6)
    # members (ARCOPY numbering), 11 output stations each
    for eid, (ni, nj, kind) in DATA["elements"].items():
        call(f"AddByPoint {eid}", m.FrameObj.AddByPoint, str(ni), str(nj), "", PROPS[kind][0], str(eid))
        call(f"SetOutputStations {eid}", m.FrameObj.SetOutputStations, str(eid), 2, 0.5, 11, False, False)
    # plane frame XZ
    call("SetActiveDOF", m.Analyze.SetActiveDOF, [True, False, True, False, True, False])
    # load patterns: the default DEAD pattern becomes SW (self-weight multiplier 1)
    call("ChangeName DEAD->SW (pattern)", m.LoadPatterns.ChangeName, "DEAD", "SW")
    call("ChangeName DEAD->SW (case)", m.LoadCases.ChangeName, "DEAD", "SW")
    call("Add G", m.LoadPatterns.Add, "G", 2, 0.0, True)                # super dead
    for c in LIVE:
        call(f"Add {c}", m.LoadPatterns.Add, c, 3, 0.0, True)          # live
    for c in ("W+", "W-"):
        call(f"Add {c}", m.LoadPatterns.Add, c, 8, 0.0, True)          # other (no automatic wind)
    # distributed loads (SW comes from the self weight); gravity dir 10, global X dir 4
    for case, lst in DATA["loads"].items():
        if case == "SW":
            continue
        for eid, direction, val in lst:
            if direction == "vertical":
                call(f"Load {case} e{eid}", m.FrameObj.SetLoadDistributed, str(eid), case, 1, 10, 0.0, 1.0,
                     -val, -val, "Global", True, True)
            else:
                call(f"Load {case} e{eid}", m.FrameObj.SetLoadDistributed, str(eid), case, 1, 4, 0.0, 1.0,
                     val, val, "Global", True, True)
    # combinations (same names and factors as ARCOPY)
    for name, (ctype, items) in COMBOS.items():
        call(f"Combo {name}", m.RespCombo.Add, name, ctype)
        for cname, is_combo, sf in items:
            call(f"Combo {name} + {cname}", m.RespCombo.SetCaseList, name, 1 if is_combo else 0, cname, sf)
    # mass: elements + G + 0.3 x live
    call("SetMassSource", m.SourceMass.SetMassSource, "ARCOPY_MASS", True, False, True, True, 5,
         ["G"] + LIVE, [1.0] + [0.3] * 4)
    # modal and response-spectrum cases
    call("SetNumberModes", m.LoadCases.ModalEigen.SetNumberModes, "MODAL", 6, 1)
    T, Sd = DATA["spectrum_T"], DATA["spectrum_Sd"]
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


def main():
    os.makedirs(OUT, exist_ok=True)
    sap = start_sap()
    m = sap.SapModel
    t0 = time.time()
    try:
        for label, crack in (("static", 1.0), ("seismic", DATA["crack"])):
            print(f"Building TestPortal_{label} ...")
            LOG.append(f"===== TestPortal_{label} (I33 factor {crack}) =====")
            build(m, crack)
            path = os.path.join(OUT, f"TestPortal_{label}.sdb")
            call("Save", m.File.Save, path)
            call("SetRunCaseFlag all", m.Analyze.SetRunCaseFlag, "", True, True)
            print("  running ...")
            call("RunAnalysis", m.Analyze.RunAnalysis)
            m.SetPresentUnits(6)
            if label == "static":
                export(m, os.path.join(OUT, "results_static"),
                       ["Joint Displacements", "Joint Reactions", "Element Forces - Frames", "Base Reactions"],
                       STATIC_CASES, list(COMBOS))
            else:
                export(m, os.path.join(OUT, "results_seismic"),
                       ["Modal Periods And Frequencies", "Modal Participating Mass Ratios", "Joint Displacements",
                        "Base Reactions", "Element Forces - Frames", "Assembled Joint Masses"],
                       ["MODAL", "RSA_X"], [])
            for tkey in ("Active Degrees of Freedom", "Frame Section Properties 01 - General",
                         "Frame Property Modifiers", "Combination Definitions", "Mass Source",
                         "Function - Response Spectrum - User", "Frame Loads - Distributed"):
                export(m, os.path.join(OUT, f"model_check_{label}"), [tkey], [], [])
    finally:
        with open(os.path.join(OUT, "build_log.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(LOG))
        sap.ApplicationExit(False)
    n_fail = sum(1 for line in LOG if line.startswith(("FAIL", "ERROR")))
    print(f"\nDone in {time.time() - t0:.0f} s, {n_fail} call(s) failed (see build_log.txt). Results in: {OUT}")


if __name__ == "__main__":
    main()
