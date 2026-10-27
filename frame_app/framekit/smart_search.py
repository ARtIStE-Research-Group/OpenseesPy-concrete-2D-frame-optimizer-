"""
Smart shear-wall search: Bayesian optimization that learns from previous trials.

Each trial is one complete retrofit strategy (wall layout -> EC8 seismic
action -> EC2 checks -> iterative jacketing -> cost), evaluated with exactly
the same routine as the screening optimizer. The Tree-structured Parzen
Estimator (TPE, Optuna) builds a probabilistic model of "good" and "bad"
layouts from all trials done so far and proposes the next layout where an
improvement is most likely. Feasibility (drift, θ, member utilization,
jacketing limits) is handled as constraints, so infeasible layouts teach the
model where not to go.

Search variables (all discrete):
    n_walls            number of wall segments (0 = no new wall)
    bay_k, start_k, len_k   bay and storey range of segment k
    tw                 wall thickness [mm]
The first trials are physics-guided seeds (walls at the failing storeys,
full-height walls in the middle/end bays, no wall). Whenever TPE proposes a
layout that was already analysed, the next trial is a local refinement of
the best strategy so far (thickness ±1 step, one storey more/less, shift by
one bay, remove or add a segment). The search stops at the budget of
complete analyses or when nothing new is proposed (converged).
"""
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from framekit.retrofit_optimizer import (
    ShearWallCandidate, optimize_global_retrofit_strategy, _walls_from_segments,
    _segment_candidate_id, _segment_description,
)


def _normalize_segments(segs: List[Tuple[int, int, int]]):
    """Merge overlapping/adjacent segments in the same bay; returns sorted tuple."""
    by_bay: Dict[int, set] = {}
    for bay, s0, s1 in segs:
        by_bay.setdefault(int(bay), set()).update(range(int(s0), int(s1) + 1))
    out = []
    for bay, stories in sorted(by_bay.items()):
        st = sorted(stories)
        start = prev = st[0]
        for s in st[1:]:
            if s == prev + 1:
                prev = s
                continue
            out.append((bay, start, prev))
            start = prev = s
        out.append((bay, start, prev))
    return tuple(out)


def _layout_params(segs, tw, max_segments):
    """Optuna parameter dict that reproduces a given layout."""
    p = {"n_walls": len(segs), "tw": float(tw)}
    for k in range(max_segments):
        bay, s0, s1 = segs[k] if k < len(segs) else (1, 1, 1)
        p[f"bay{k}"], p[f"start{k}"], p[f"len{k}"] = int(bay), int(s0), int(s1 - s0 + 1)
    return p


def _neighbours(segs, tw, geom, thicknesses, max_segments):
    """Layouts one small step away: thickness ±1 step, a segment one storey
    longer/shorter (at either end), shifted by one bay, removed, or a new
    one-storey segment added."""
    nf, nb = int(geom.num_floors), int(geom.num_bays)
    th = sorted(thicknesses)
    out = []
    i = th.index(tw) if tw in th else 0
    for j in (i - 1, i + 1):
        if 0 <= j < len(th) and segs:
            out.append((list(segs), th[j]))
    for k, (b, s0, s1) in enumerate(segs):
        others = [x for m, x in enumerate(segs) if m != k]
        for nb_, ns0, ns1 in ((b, s0 - 1, s1), (b, s0, s1 + 1), (b, s0 + 1, s1), (b, s0, s1 - 1),
                              (b - 1, s0, s1), (b + 1, s0, s1)):
            if 1 <= nb_ <= nb and 1 <= ns0 <= ns1 <= nf:
                out.append((others + [(nb_, ns0, ns1)], tw))
        out.append((others, tw))
    if len(segs) < max_segments:
        for b in range(1, nb + 1):
            out.append((list(segs) + [(b, 1, 1)], tw))
    res = []
    for sg, t in out:
        sg = list(_normalize_segments(sg)) if sg else []
        if len(sg) <= max_segments:
            res.append((sg, t))
    return res


def _physics_seeds(geom, failed_stories, thicknesses, max_segments):
    nf, nb = int(geom.num_floors), int(geom.num_bays)
    mid = int(math.ceil(nb / 2.0))
    tw = 200.0 if 200.0 in thicknesses else float(thicknesses[len(thicknesses) // 2])
    fs = sorted(int(s) for s in (failed_stories or []) if 1 <= int(s) <= nf)

    def params(segs):
        return _layout_params(segs, tw, max_segments)

    seeds = [params([]), params([(mid, 1, nf)])]
    if fs:
        seeds.append(params([(mid, min(fs), max(fs))]))
        seeds.append(params([(mid, 1, max(fs))]))
    if max_segments >= 2 and nb >= 2:
        seeds.append(params([(1, 1, nf), (nb, 1, nf)]))
    # unique
    out, seen = [], set()
    for s in seeds:
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def smart_wall_search(*, global_kwargs: Dict[str, Any], geom, base_walls=None, wall_E_Pa: float,
                      thicknesses_mm=(150.0, 200.0, 250.0, 300.0), n_trials: int = 60, max_segments: int = 2,
                      failed_stories: Optional[List[int]] = None, seed: int = 42, n_startup_trials: int = 10,
                      verbose: bool = True):
    """Bayesian (TPE) search of complete retrofit strategies.

    global_kwargs: the keyword arguments normally passed to
    optimize_global_retrofit_strategy (model, combinations, costs, seismic
    settings, limits). Returns (global_opt_df, history_df) with
    global_opt_df in the same format as the screening optimizer.
    """
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    nf, nb = int(geom.num_floors), int(geom.num_bays)
    thicknesses = [float(t) for t in thicknesses_mm]
    base_panels = {(int(w.bay_left), int(w.floor_top)) for w in (base_walls or [])}
    cache: Dict[Any, Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []
    best_cost = [np.inf]
    t_start = time.time()

    kw = dict(global_kwargs)
    kw.pop("candidates_override", None)
    kw.pop("include_no_new_wall_candidate", None)
    kw["max_global_candidates"] = None

    def layout_key(segs, tw):
        panels = sorted((b, s) for b, s0, s1 in segs for s in range(s0, s1 + 1))
        return tuple(panels), (float(tw) if panels else 0.0)

    def evaluate(segs, tw):
        panels = sorted((b, s) for b, s0, s1 in segs for s in range(s0, s1 + 1))
        key = layout_key(segs, tw)
        if key in cache:
            return cache[key], True
        if any(p in base_panels for p in panels):
            cache[key] = None
            return None, False
        if segs:
            walls = _walls_from_segments(segs)
            for w in walls:
                w.tw_mm = tw
            cand = ShearWallCandidate(
                candidate_id=_segment_candidate_id(segs, tw), walls=walls, wall_tw_mm=tw, wall_E_Pa=float(wall_E_Pa),
                description=f"Wall segment(s): {_segment_description(segs)}; smart search, tw={tw:.0f} mm")
            df = optimize_global_retrofit_strategy(**kw, candidates_override=[cand],
                                                   include_no_new_wall_candidate=False)
        else:
            df = optimize_global_retrofit_strategy(**kw, candidates_override=[],
                                                   include_no_new_wall_candidate=True)
        row = df.iloc[0].to_dict() if df is not None and not df.empty else None
        cache[key] = row
        return row, False

    def _set_constraint(trial, values):
        trial.set_user_attr("constraint", list(values))
        if hasattr(trial, "set_constraint"):          # Optuna >= 5
            for name, v in zip(("drift", "member_utilization", "complete_feasibility"), values):
                trial.set_constraint(name, float(v))

    def objective(trial):
        n = trial.suggest_int("n_walls", 0, max_segments)
        segs = []
        for k in range(max_segments):
            bay = trial.suggest_int(f"bay{k}", 1, nb)
            s0 = trial.suggest_int(f"start{k}", 1, nf)
            ln = trial.suggest_int(f"len{k}", 1, nf)
            if k < n:
                segs.append((bay, s0, min(nf, s0 + ln - 1)))
        tw = float(trial.suggest_categorical("tw", thicknesses))
        segs = list(_normalize_segments(segs)) if segs else []
        t0 = time.time()
        row, cached = evaluate(segs, tw)
        if row is None:
            _set_constraint(trial, [1.0, 1.0, 1.0])
            history.append({"Trial": trial.number + 1, "Layout": _segment_description(segs) or "no new wall",
                            "Wall_tw_mm": tw, "Status": "invalid (overlaps existing wall)", "Cached": cached})
            return 1e12
        cost = float(row.get("Total_Retrofit_Cost", np.nan))
        drift_u = float(pd.to_numeric(row.get("Drift_Utilization"), errors="coerce"))
        eta = float(pd.to_numeric(row.get("Final_Max_EC2_Utilization"), errors="coerce"))
        feasible = str(row.get("Complete_Feasible", "")).upper() == "YES"
        # constraints (feasible when all <= 0): drift utilization, member
        # utilization, and the complete-feasibility flag (theta, jacketing limits)
        _set_constraint(trial, [
            (drift_u - 1.0) if np.isfinite(drift_u) else 1.0,
            (eta - 1.0) if np.isfinite(eta) else 1.0,
            -1.0 if feasible else 1.0,
        ])
        if feasible and cost < best_cost[0]:
            best_cost[0] = cost
        history.append({
            "Trial": trial.number + 1, "Analysis": sum(r is not None for r in cache.values()),
            "Layout": _segment_description(segs) or "no new wall",
            "Wall_tw_mm": tw if segs else 0.0, "Num_Panels": sum(s1 - s0 + 1 for _, s0, s1 in segs),
            "Total_Cost": cost, "Wall_Cost": row.get("Wall_Cost"), "Jacketing_Cost": row.get("Jacketing_Cost"),
            "Drift_Utilization": drift_u, "Max_Eta": eta, "Feasible": "YES" if feasible else "NO",
            "Best_Feasible_Cost_So_Far": best_cost[0] if np.isfinite(best_cost[0]) else np.nan,
            "Cached": cached, "Eval_Time_s": time.time() - t0,
            "Source": trial.user_attrs.get("source") or
                      ("random start" if trial.number < n_startup_trials else "TPE (learned)"),
            "_segments": segs,
        })
        if verbose:
            best_txt = f"{best_cost[0]:,.0f}" if np.isfinite(best_cost[0]) else "–"
            n_done = sum(r is not None for r in cache.values())
            print(f"[smart {n_done:3d}/{n_trials} | trial {trial.number + 1:3d}] {history[-1]['Layout']:<24s} t={tw:3.0f} "
                  f"cost {cost:>10,.0f}  {'FEASIBLE' if feasible else 'not feasible':<12s} best {best_txt}"
                  + ("  (cached)" if cached else ""))
        return cost

    import warnings
    tpe_kw = dict(seed=seed, n_startup_trials=n_startup_trials, multivariate=True)
    if not hasattr(optuna.trial.Trial, "set_constraint"):   # Optuna 3.x / 4.x
        tpe_kw["constraints_func"] = lambda t: t.user_attrs.get("constraint", [1.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = TPESampler(**tpe_kw)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for p in _physics_seeds(geom, failed_stories, thicknesses, max_segments):
        study.enqueue_trial(p, user_attrs={"source": "physics seed"})
    print(f"\nSmart search: budget {n_trials} complete analyses (physics seeds, then {n_startup_trials} "
          f"exploratory trials, then TPE learning from all previous trials).")
    # the budget counts complete analyses: repeated layouts come from the
    # cache at no cost and do not use up the budget
    # the search also stops when it keeps proposing layouts already analysed
    # (it has converged on the best region)
    patience = max(20, int(n_trials) // 2)
    converged = [False]

    queued = set()

    def _best_layout():
        h = [x for x in history if "Total_Cost" in x and np.isfinite(x.get("Total_Cost", np.nan))]
        hf = [x for x in h if x.get("Feasible") == "YES"]
        pool = hf or h
        if not pool:
            return None
        b = min(pool, key=lambda x: x["Total_Cost"])
        return b["_segments"], float(b["Wall_tw_mm"]) if b["_segments"] else float(thicknesses[0])

    def _stop(st, _tr):
        if sum(r is not None for r in cache.values()) >= int(n_trials):
            st.stop()
            return
        # when TPE proposes a layout already analysed, the next trial is a
        # local refinement around the best strategy found so far
        if history and history[-1].get("Cached"):
            bl = _best_layout()
            if bl is not None:
                for sg, t in _neighbours(bl[0], bl[1], geom, thicknesses, max_segments):
                    k = layout_key(sg, t)
                    if k in cache or k in queued:
                        continue
                    queued.add(k)
                    st.enqueue_trial(_layout_params(sg, t, max_segments),
                                     user_attrs={"source": "local refinement"})
                    return
        tail = history[-patience:]
        if len(tail) >= patience and all(bool(h.get("Cached")) for h in tail):  # no new neighbour either
            converged[0] = True
            st.stop()
    study.optimize(objective, n_trials=int(n_trials) * 5, callbacks=[_stop])

    hist = pd.DataFrame(history)
    rows = [r for r in cache.values() if r is not None]
    g = pd.DataFrame(rows)
    if not g.empty:
        g["_feas"] = g["Complete_Feasible"].astype(str).str.upper().eq("YES")
        g = g.sort_values(["_feas", "Total_Retrofit_Cost"], ascending=[False, True]).drop(columns="_feas")
        g = g.reset_index(drop=True)
        g["Global_Rank"] = np.arange(1, len(g) + 1)
        g["Selected"] = ["YES"] + ["NO"] * (len(g) - 1)
        g["Search_Method"] = "Bayesian optimization (TPE)"
    if converged[0]:
        print(f"Smart search converged: the last {patience} proposals were layouts already analysed.")
    print(f"Smart search finished: {len(hist)} trials, {len(rows)} unique strategies analysed in "
          f"{(time.time() - t_start) / 60:.1f} min.")
    return g, hist


# ============================================================
# plots of the learning process
# ============================================================
def _valid(hist, unique=True):
    """Trials with a result; unique=True keeps only new analyses (no cache hits)."""
    if hist is None or getattr(hist, "empty", True) or "Total_Cost" not in hist:
        return None
    h = hist[pd.to_numeric(hist["Total_Cost"], errors="coerce").notna()].copy()
    if unique and "Cached" in h:
        h = h[~h["Cached"].astype(bool)]
    return h


def plot_learning_curve(hist, ax=None):
    """Cost of every trial and best feasible cost found so far."""
    import matplotlib.pyplot as plt
    h = _valid(hist)
    if h is None or h.empty:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.6))
    feas = h["Feasible"].eq("YES")
    styles = {"physics seed": ("D", "#2A9D8F"), "random start": ("o", "#9AA5B1"), "TPE (learned)": ("o", "#1D3557"),
              "local refinement": ("^", "#F4A261")}
    for src, (mk, col) in styles.items():
        m = h["Source"].eq(src)
        if (m & feas).any():
            ax.scatter(h.loc[m & feas, "Analysis"], h.loc[m & feas, "Total_Cost"], marker=mk, s=30, color=col,
                       label=f"{src} (feasible)", zorder=3)
        if (m & ~feas).any():
            ax.scatter(h.loc[m & ~feas, "Analysis"], h.loc[m & ~feas, "Total_Cost"], marker=mk, s=30,
                       facecolor="none", edgecolor=col, label=f"{src} (not feasible)", zorder=3)
    best = pd.to_numeric(h["Best_Feasible_Cost_So_Far"], errors="coerce")
    if best.notna().any():
        ax.step(h["Analysis"], best, where="post", color="#E63946", lw=2.2, label="Best feasible so far", zorder=4)
    # y-range covers every analysed trial (no points hidden above the axis);
    # penalised placeholders of failed analyses (>= 1e11) are left out
    c = pd.to_numeric(h["Total_Cost"], errors="coerce")
    c = c[np.isfinite(c) & (c < 1e11)]
    if not c.empty:
        ax.set_ylim(0.97 * c.min(), 1.03 * c.max())
    ax.set_xlabel("Complete analysis no. (repeated layouts are not re-analysed)")
    ax.set_ylabel("Total retrofit cost [EUR]")
    ax.set_title("Smart search: learning curve", loc="center", fontsize=11)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    return ax


def plot_search_map(hist, geom, best_segments=None, ax=None):
    """How often each wall panel (bay × storey) was tried; outline = selected layout."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    h = _valid(hist)
    if h is None or h.empty or "_segments" not in h:
        return None
    nf, nb = int(geom.num_floors), int(geom.num_bays)
    cnt = np.zeros((nf, nb))
    feas_cnt = np.zeros((nf, nb))
    for segs, f in zip(h["_segments"], h["Feasible"]):
        for b, s0, s1 in segs or []:
            for s in range(s0, s1 + 1):
                cnt[s - 1, b - 1] += 1
                feas_cnt[s - 1, b - 1] += f == "YES"
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * nb + 3, 0.8 * nf + 2))
    im = ax.imshow(cnt / max(len(h), 1), origin="lower", cmap="Blues", vmin=0, aspect="auto",
                   extent=(0.5, nb + 0.5, 0.5, nf + 0.5))
    for s in range(nf):
        for b in range(nb):
            if cnt[s, b] > 0:
                ax.text(b + 1, s + 1, f"{int(cnt[s, b])}\n({int(feas_cnt[s, b])} ok)", ha="center", va="center",
                        fontsize=7.5, color="white" if cnt[s, b] / len(h) > 0.5 else "#222222")
    for b, s0, s1 in best_segments or []:
        ax.add_patch(Rectangle((b - 0.5, s0 - 0.5), 1, s1 - s0 + 1, fill=False, ec="#E63946", lw=3))
    ax.set_xticks(range(1, nb + 1))
    ax.set_yticks(range(1, nf + 1))
    ax.set_xlabel("Bay")
    ax.set_ylabel("Storey")
    ax.set_title("Smart search map: share of analysed strategies with a wall in each panel\n(red outline = selected layout)",
                 loc="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="share of analysed strategies")
    return ax


def show_smart_search_plots(hist, geom, global_opt_df=None):
    import matplotlib.pyplot as plt
    if _valid(hist) is None:
        return
    best_segs = best_segments_from_history(hist, global_opt_df)
    plot_learning_curve(hist)
    plt.show()
    plot_search_map(hist, geom, best_segs)
    plt.show()


def best_segments_from_history(hist, global_opt_df):
    """Segments of the selected strategy (cheapest feasible trial, else cheapest)."""
    h = _valid(hist, unique=False)
    if h is None or h.empty:
        return None
    hf = h[h["Feasible"].eq("YES")]
    hh = hf if not hf.empty else h
    return hh.sort_values("Total_Cost").iloc[0]["_segments"]
