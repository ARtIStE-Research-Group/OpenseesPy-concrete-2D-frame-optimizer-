"""Console answers (input files) for the six frames of the ARCOPY paper (Section 4).

    python make_inputs.py            writes the files in inputs/

kind = "base"   : assessment of the existing frame only (no retrofit); the SAP2000 models of
                  verification/sap2000 are built from this run
kind = "screen" : retrofit with the drift-based screening (N = 40)
kind = "bayes"  : retrofit with the Bayesian search (budget of 40 complete analyses)
"bayes3"        : Bayesian search with up to three wall segments per layout (eight-storey frames)

All runs: existing building (EN 1998-3, KL2), transformed sections, EI x 0.5 in the seismic analysis,
NTC 2008 spectrum with q = 1.5, alpha = 0.005, nu = 0.5, walls 150-300 mm, jacket thickness up to
200 mm, added steel ratio up to 1.5, default unit costs.
Wind: 1.25, 3.75, 6.25, ... kN/m (+2.5 kN/m per storey) on the windward column line, W- mirrored.
"""
B5 = [5, 5, 6, 5, 5]
B7 = [4, 5, 6, 5, 6, 5, 4]
FRAMES = {
    "F4-5B": dict(storeys=4, bays=B5, beam=(300, 400), col=(400, 400)),
    "F6-5B": dict(storeys=6, bays=B5, beam=(400, 500), col=(500, 500)),
    "F8-5B": dict(storeys=8, bays=B5, beam=(400, 500), col=(500, 500)),
    "F4-7B": dict(storeys=4, bays=B7, beam=(300, 400), col=(400, 400)),
    "F6-7B": dict(storeys=6, bays=B7, beam=(400, 500), col=(500, 500)),
    "F8-7B": dict(storeys=8, bays=B7, beam=(400, 500), col=(500, 500)),
}
# Reinforcement (assumed existing detailing, the same for all members of the six frames):
# columns 12 phi16 with phi8 ties at 100 mm (4 legs), beams 3 phi16 top and bottom with phi8
# stirrups at 150 mm (2 legs); covers 30 mm (beams) and 35 mm (columns).
DATA = dict(
    H=3.0, fck=25, fyk=450, E_MPa=31000, density=2500,
    G_floor=23.0, G_roof=23.0, Q=2.5,
    wind_step=2.5,             # kN/m: storey i gets wind_step * (i - 0.5) on the windward column line
    # reinforcement: beam (cover, n_top, phi_top, n_bot, phi_bot, phi_st, s, legs);
    #                column (cover, n_top, phi, n_bot, phi, n_left, n_right, phi_side, phi_st, s, legs)
    beam_rc={(300, 400): (30, 3, 16, 3, 16, 8, 150, 2), (400, 500): (30, 3, 16, 3, 16, 8, 150, 2)},
    col_rc={(400, 400): (35, 4, 16, 4, 16, 2, 2, 16, 8, 100, 4), (500, 500): (35, 4, 16, 4, 16, 2, 2, 16, 8, 100, 4)},
)
NTC = dict(ag=0.193493, F0=2.443832, Tc=0.33327, soil="B", topo="T1")


def answers(name, kind, budget=40, N=40, run_name=None, max_walls=""):
    f = FRAMES[name]
    d = DATA
    ns, bays = f["storeys"], f["bays"]
    a = [run_name or f"{name}_{kind}", "rc", ns, d["H"], len(bays), *bays, "n", "3"]
    bm = d["beam_rc"][f["beam"]]
    a += [f["beam"][0], f["beam"][1], *bm, "n"]
    c = d["col_rc"][f["col"]]
    a += [f["col"][0], f["col"][1], *c, "n"]
    a += ["2", d["fck"], d["fyk"], "n", "n", "n", "2", "KL2", "n", d["E_MPa"], d["density"]]
    a += ["1"]                       # section stiffness: gross / transformed
    a += [d["G_floor"]] * (ns - 1) + [d["G_roof"]]
    mid = len(bays) // 2 + 1
    a += [1, "", d["Q"], 1, mid, "n"]                            # live01 = Q on the middle bay of floor 1
    a += ["", *[d["wind_step"] * (i - 0.5) for i in range(1, ns + 1)], "n", "", "n"]          # wind per storey, no leeward part, W- mirrored
    a += ["n", "Q_LIN", "1", "", "", "", "", ""]                 # Q as a Linear Add group (as in SAP2000)
    a += ["", "", ""]                                            # own combinations, member sections
    a += ["y", "n", "n", "3"]                                    # EC2, no tables/plots, no N/V/M plots
    a += ["y", "0", "n", "", ns, "n", "n"]
    a += ["y", "2", NTC["ag"], NTC["F0"], NTC["Tc"], NTC["soil"], NTC["topo"], "0.05", "CQC"]
    a += ["", "n", "1", ""]       # q, spectrum plot, DLS alpha 0.005, nu
    a += ["n"]                                                   # time history
    if kind == "base":
        a += ["y", "n"]                                          # performance summary, no retrofit
    else:
        a += ["y", "y", max_walls, "1" if kind == "screen" else "2"]   # max wall segments per layout, method
        a += [N] if kind == "screen" else [budget]
        a += ["", "", "", "", "", "", ""]
    return "\n".join(str(x) for x in a) + "\n"


if __name__ == "__main__":
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "inputs")
    os.makedirs(out, exist_ok=True)
    n = 0
    for name in FRAMES:
        for kind in ("base", "screen", "bayes"):
            open(os.path.join(out, f"{name}_{kind}.txt"), "w", newline="\n").write(answers(name, kind))
            n += 1
    # eight-storey frames: Bayesian search with up to three wall segments per layout
    for name in ("F8-5B", "F8-7B"):
        open(os.path.join(out, f"{name}_bayes3.txt"), "w", newline="\n").write(
            answers(name, "bayes", run_name=f"{name}_bayes3", max_walls="3"))
        n += 1
    print("written", n, "files in", out)
