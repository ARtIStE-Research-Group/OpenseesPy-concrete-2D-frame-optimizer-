"""Runs the input files of the six frames one after the other (no typing, no figure windows).

    python examples/six_frames/run_all.py                    all 20 runs (about 45 min)
    python examples/six_frames/run_all.py F4-5B_bayes        only this run
    python examples/six_frames/run_all.py --jobs 2           two runs at the same time

Each run is written to a folder with the name of the input file, next to main.py; an existing
folder with that name is deleted first. The console output goes to examples/six_frames/logs/.
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUTS = os.path.join(HERE, "inputs")
LOGS = os.path.join(HERE, "logs")


def run(job):
    t0 = time.time()
    shutil.rmtree(os.path.join(ROOT, job), ignore_errors=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    with open(os.path.join(INPUTS, job + ".txt")) as fin, \
            open(os.path.join(LOGS, job + ".log"), "w", encoding="utf-8") as fout:
        code = subprocess.call([sys.executable, "-u", "main.py"], cwd=ROOT, stdin=fin, stdout=fout,
                               stderr=subprocess.STDOUT, env=env)
    print(f"{job:14s} finished in {(time.time() - t0) / 60:5.1f} min (return code {code})", flush=True)
    return code


if __name__ == "__main__":
    args = sys.argv[1:]
    jobs = 1
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i:i + 2]
    names = args or sorted(f[:-4] for f in os.listdir(INPUTS) if f.endswith(".txt"))
    os.makedirs(LOGS, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        codes = list(ex.map(run, names))
    sys.exit(max(codes) if codes else 0)
