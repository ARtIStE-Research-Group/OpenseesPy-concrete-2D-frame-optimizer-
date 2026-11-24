"""Writes six_frames_data.json: everything the SAP2000 script needs, taken from the ARCOPY baseline runs
(<frame>_base/_checkpoints) so that no number is typed by hand."""
import json, os, pickle, sys
import numpy as np

ARC = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "six_frames_data.json")
sys.path.insert(0, os.path.join(ARC, "frame_app"))
from framekit.opensees_model import section_properties_from_fiber, _get_sec_attr   # noqa: E402
from framekit.ec8_seismic import design_spectrum_ordinate, _soil                   # noqa: E402

FRAMES = ["F4-5B", "F6-5B", "F8-5B", "F4-7B", "F6-7B", "F8-7B"]
data = {}
for name in FRAMES:
    ck = os.path.join(ARC, name + "_base", "_checkpoints")
    st = pickle.load(open(os.path.join(ck, "step_03_static.pkl"), "rb"))
    md = pickle.load(open(os.path.join(ck, "step_06_modal.pkl"), "rb"))
    rs = pickle.load(open(os.path.join(ck, "step_07_rsa.pkl"), "rb"))
    Ec, Es = st["Ec_Pa"], st["Es_Pa"]
    sec = {}
    for kind, s in (("beam", st["beam_sec"]), ("column", st["column_sec"])):
        b, h = float(_get_sec_attr(s, "b", "b_mm")) / 1000.0, float(_get_sec_attr(s, "h", "h_mm")) / 1000.0
        A, I = section_properties_from_fiber(s, Ec, Es)
        sec[kind] = {"b": b, "h": h, "A_mod": A / (b * h), "I33_mod": I / (b * h ** 3 / 12.0)}
    loads = {}
    for case, lst in st["cases"].items():
        if case in ("SW", "Q"):
            continue
        rows = []
        for ld in lst:
            assert ld.load_type == "uniform" and ld.element is not None, (name, case, ld)
            rows.append([int(ld.element), ld.direction, float(ld.value)])
        loads[case] = rows
    # self weight per metre of a column -> unit weight
    sw = next(l for l in st["cases"]["SW"] if st["elements"][l.element][2] == "column")
    gamma = abs(sw.value) / (sec["column"]["b"] * sec["column"]["h"])
    groups = st["load_groups"]["groups"]
    combos = {}
    for cname, c in st["combos"].items():
        if cname in groups:
            continue
        items = {}
        for k, f in c["factors"].items():
            if k in groups:
                assert groups[k]["type"] == "Linear Add"
                for mem in groups[k]["members"]:
                    items[mem] = items.get(mem, 0.0) + float(f)
            else:
                items[k] = items.get(k, 0.0) + float(f)
        combos[cname] = [[k, round(v, 10)] for k, v in items.items()]
    s = _soil(rs["rsa_params"])
    T = sorted(set(np.round(np.arange(0.0, 4.0001, 0.005), 3).tolist() + [s["TB"], s["TC"], s["TD"]]))
    Sd = [float(design_spectrum_ordinate(t, rs["rsa_params"], rs["q_beh"])) for t in T]
    assert not md["use_diaphragm"] and not st["walls"]
    data[name] = {
        "nodes": {str(k): [float(v[0]), float(v[1])] for k, v in st["nodes"].items()},
        "elements": {str(k): [int(v[0]), int(v[1]), v[2]] for k, v in st["elements"].items()},
        "sections": sec, "E_kPa": Ec / 1000.0, "unit_weight": gamma, "loads": loads, "combos": combos,
        "crack": float(md["crack_factor"]), "n_modes": int(md["nmodes"]),
        "floor_mass_t": {str(k): float(v) / 1000.0 for k, v in rs["floor_masses"].items()},
        "storey_height": float(st["geom"].floor_height),
        "spectrum_T": T, "spectrum_Sd": Sd, "q": float(rs["q_beh"]),
    }
    print(name, len(st["nodes"]), "nodes", len(st["elements"]), "members", sec, round(gamma, 4), data[name]["crack"],
          data[name]["n_modes"], list(combos))
json.dump(data, open(OUT, "w"))
print("written", OUT, os.path.getsize(OUT))
