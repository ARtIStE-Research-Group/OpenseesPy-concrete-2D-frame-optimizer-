"""
Automatic step checkpoints for the ARCOPY main workflow.

After every completed step, the full workflow state (all user inputs and
results created so far) is saved to disk. When main.py is started again it
detects the saved steps and lets the user resume after the last completed
step, or restart from any earlier step, without re-entering the model.
"""
import os
import sys
import time
import json
import pickle
import shutil
import types


STEP_DESCRIPTIONS = {
    "model": "Geometry, shear walls, sections, materials",
    "loads": "Load cases and Eurocode combinations",
    "static": "Static analysis of all combinations + drifts",
    "ec2": "EC2 member checks",
    "nvm": "N/V/M diagrams",
    "modal": "Seismic masses + eigen analysis",
    "rsa": "EC8 response spectrum + seismic verifications",
    "th": "Linear time-history analysis",
    "perf": "Global performance summary",
    "retrofit_search": "Retrofit optimization (search)",
    "retrofit_select": "Retrofit optimization (selected strategy)",
    "report": "Retrofit report",
}


class WorkflowCheckpoint:
    def __init__(self, folder, steps, namespace):
        self.folder = os.path.abspath(folder)
        self.steps = list(steps)
        self.ns = namespace
        self.status_path = os.path.join(self.folder, "status.json")
        self.skip = set()
        self.current = None
        # Names that exist before the workflow starts (imports, helpers) are
        # not saved - unless the workflow assigns them a new value. (Spyder can
        # run main.py in a console that still holds the variables of a
        # previous run: those are saved as soon as this run recomputes them.)
        self._baseline = dict(namespace)   # keeps references: compared by identity

    # --------------------------------------------------------
    # files
    # --------------------------------------------------------
    def _snap_path(self, step):
        idx = self.steps.index(step) + 1
        return os.path.join(self.folder, f"step_{idx:02d}_{step}.pkl")

    def _read_status(self):
        try:
            with open(self.status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_status(self, **kw):
        st = self._read_status()
        st.update(kw)
        st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(self.folder, exist_ok=True)
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        os.replace(tmp, self.status_path)

    def _completed_steps(self):
        return [s for s in self.steps if os.path.exists(self._snap_path(s))]

    # --------------------------------------------------------
    # state capture
    # --------------------------------------------------------
    def _collect_state(self):
        state, skipped = {}, []
        for k, v in list(self.ns.items()):
            if k.startswith("__") or (k in self._baseline and self._baseline[k] is v):
                continue
            if isinstance(v, (types.ModuleType, types.FunctionType, type, WorkflowCheckpoint)):
                continue
            try:
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                skipped.append(k)
                continue
            state[k] = v
        return state, skipped

    def _load_snapshot(self, step):
        with open(self._snap_path(step), "rb") as f:
            state = pickle.load(f)
        self.ns.update(state)

    # --------------------------------------------------------
    # public API
    # --------------------------------------------------------
    def start(self):
        done = self._completed_steps()
        if not done:
            self._write_status(running=None, last_error_step=None)
            print(f"\n[checkpoint] Progress is saved automatically after every step in:\n  {self.folder}")
            return

        st = self._read_status()
        print("\n" + "=" * 62)
        print(" Saved ARCOPY session found" + (f"  (last update {st.get('updated')})" if st.get("updated") else ""))
        print("=" * 62)
        for i, s in enumerate(self.steps, start=1):
            mark = "✔" if s in done else " "
            print(f"  {mark} {i:2d}) {s:16s} {STEP_DESCRIPTIONS.get(s, '')}")
        if st.get("running"):
            print(f"\n  ⚠ The previous run stopped during step: {st['running']}")

        # contiguous completed prefix
        prefix = 0
        for s in self.steps:
            if s in done:
                prefix += 1
            else:
                break
        next_step = self.steps[prefix] if prefix < len(self.steps) else None

        print("\nOptions:")
        if next_step:
            print(f"  [Enter]  resume from step {prefix + 1} ({next_step})")
        else:
            print("  [Enter]  all steps completed - reload results (nothing re-run)")
        print(f"  1..{min(prefix + 1, len(self.steps))}     restart from that step (earlier steps are reused)")
        print("  n        start a NEW model (old checkpoints are archived)")

        while True:
            ans = input("Choice: ").strip().lower()
            if ans == "":
                start_idx = prefix + 1
                break
            if ans == "n":
                self._archive()
                self._write_status(running=None)
                print("[checkpoint] Starting a new model.")
                return
            try:
                k = int(ans)
            except ValueError:
                print("❌ Enter a step number, n, or press Enter.")
                continue
            if 1 <= k <= min(prefix + 1, len(self.steps)):
                start_idx = k
                break
            print(f"❌ Choose between 1 and {min(prefix + 1, len(self.steps))}.")

        # remove snapshots from start_idx onward
        for s in self.steps[start_idx - 1:]:
            p = self._snap_path(s)
            if os.path.exists(p):
                os.remove(p)

        if start_idx > 1:
            self._load_snapshot(self.steps[start_idx - 2])
        self.skip = set(self.steps[: start_idx - 1])
        if start_idx <= len(self.steps):
            print(f"[checkpoint] Restored state. Continuing from step {start_idx} ({self.steps[start_idx - 1]}).")
        else:
            print("[checkpoint] Restored state of the completed session.")

    def need(self, step):
        """True if this step has to be executed now."""
        if step in self.skip:
            return False
        self.current = step
        self._write_status(running=step)
        print("\n" + "─" * 62)
        print(f" STEP {self.steps.index(step) + 1}: {STEP_DESCRIPTIONS.get(step, step)}")
        print("─" * 62)
        return True

    def done(self, step):
        state, skipped = self._collect_state()
        os.makedirs(self.folder, exist_ok=True)
        path = self._snap_path(step)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        self._write_status(running=None, last_done=step)
        msg = f"[checkpoint] ✔ step '{step}' saved"
        if skipped:
            msg += f" (not saved: {', '.join(skipped)})"
        print(msg)

    def finish(self):
        self._write_status(running=None, finished=True)
        print(f"\n[checkpoint] Workflow finished. Saved steps are kept in:\n  {self.folder}")

    def _archive(self):
        if not os.path.isdir(self.folder):
            return
        files = [f for f in os.listdir(self.folder) if f.endswith(".pkl") or f == "status.json"]
        if not files:
            return
        dst = os.path.join(self.folder, "archive_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.move(os.path.join(self.folder, f), os.path.join(dst, f))
        print(f"[checkpoint] Previous session archived in: {dst}")
