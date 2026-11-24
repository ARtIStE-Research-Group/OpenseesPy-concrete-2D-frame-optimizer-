"""
Redraw the N/V/M diagrams of a saved run, with all its inputs, from the checkpoints.

Nothing is asked again and the checkpoints are not changed: the script only reads
the saved model and load combinations and draws the diagrams with the current
code. The figures are saved in <run folder>/figures_nvm/ (and shown in Spyder).

Spyder: write the run folder name in RUN below (e.g. "my_run") and press Run.
Terminal: python redraw_nvm.py my_run                 (all combinations)
          python redraw_nvm.py my_run CROWD WIND       (only these)
"""
import os
import pickle
import sys

RUN = ""            # run folder name, e.g. "my_run" (empty = ask)
COMBOS = []         # combination names, e.g. ["CROWD", "WIND"] (empty = ask; Enter = all)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frame_app"))

# Spyder keeps modules in memory: use the framekit of this folder
for _m in [m for m in list(sys.modules) if m == "framekit" or m.startswith("framekit.")]:
    del sys.modules[_m]

from framekit.plot_style import apply_style, enable_figure_autosave  # noqa: E402
from framekit.diagrams import plot_all_combos_NVM  # noqa: E402
from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static  # noqa: E402
from framekit.member_sections import set_member_sections  # noqa: E402

# the model as it was before the retrofit (the N/V/M step of the workflow)
SNAPSHOTS = ["step_05_nvm.pkl", "step_04_ec2.pkl", "step_03_static.pkl"]


def main(run=None, combos=None):
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    run = run or RUN or (args[0] if args else "") or input("Run folder name: ").strip()
    run_dir = run if os.path.isabs(run) else os.path.join(HERE, run)
    ck_dir = os.path.join(run_dir, "_checkpoints")
    snap = next((os.path.join(ck_dir, f) for f in SNAPSHOTS if os.path.exists(os.path.join(ck_dir, f))), None)
    if snap is None:
        raise SystemExit(f"No checkpoint after the static analysis in {ck_dir}: run main.py at least to step 3.")
    with open(snap, "rb") as f:
        ck = pickle.load(f)
    print(f"Model and combinations read from {os.path.relpath(snap, HERE)}")

    names = list(ck["names"])
    wanted = list(combos or COMBOS or args[1:])
    if not wanted:
        for i, c in enumerate(names, start=1):
            print(f"{i:3d}) {c}")
        raw = input("Combinations to draw (numbers or names separated by commas, Enter = all): ").strip()
        for tok in [t.strip() for t in raw.split(",") if t.strip()]:
            wanted.append(names[int(tok) - 1] if tok.isdigit() else tok)
    wanted = wanted or names
    unknown = [c for c in wanted if c not in ck["combos"]]
    if unknown:
        raise SystemExit(f"Unknown combination(s): {', '.join(unknown)}")

    set_member_sections(ck.get("n_sections_beam"), ck.get("n_sections_column"))
    apply_style()
    out = os.path.join(run_dir, "figures_nvm")
    enable_figure_autosave(out)
    plot_all_combos_NVM(ck["nodes"], ck["elements"], ck["combos"], wanted, ck["Ec_Pa"], ck["Es_Pa"],
                        ck["column_sec"], ck["beam_sec"], ck["geom"], ops_build_model, ops_apply_combo_loads,
                        ops_run_static, walls=ck.get("walls"), wall_tw_mm=ck.get("wall_tw_mm", 200.0),
                        wall_E_Pa=ck.get("wall_E_Pa", 30e9), npts=60)
    print(f"\n✔ {len(wanted)} combination(s) drawn; figures saved in {out}")


if __name__ == "__main__":
    main()
