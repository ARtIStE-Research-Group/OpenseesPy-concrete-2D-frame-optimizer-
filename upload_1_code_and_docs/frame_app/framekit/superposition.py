"""Result-level load combinations (Range Add, Envelope) by linear superposition.

A combination that contains a Range Add or Envelope group cannot be written as
one set of loads. ARCOPY then analyses each basic load case once (linear
analysis) and combines the results, as SAP2000 does:

  Linear Add  r = sum of the members
  Range Add   max = sum of the positive parts of the members,
              min = sum of the negative parts (each member on only where it
              makes the result larger, or smaller)
  Envelope    max / min over the members

The combination itself is factors x (cases and groups). Every response
(end forces, N/V/M at the sections, storey drifts) gets its own maximum and
minimum, as in SAP2000 envelopes. The checks use these extremes, like the
maxima of N, V and M that they already use for ordinary combinations.

`GroupLoads` is a list of loads (all members on, used only to draw the loads)
that also carries the definition of the combination.
"""
from collections import OrderedDict

import numpy as np


class GroupLoads(list):
    """Loads of a result-level combination.

    terms  : [(factor, item)], item = basic load case or group name
    groups : {name: {"type": "Linear Add" | "Range Add" | "Envelope", "members": [...]}}
    cases  : {case: [Load, ...]} basic load cases used by the combination
    """

    def __init__(self, loads=(), terms=(), groups=None, cases=None):
        super().__init__(loads)
        self.terms = [(float(f), str(it)) for f, it in terms]
        self.groups = {k: {"type": v["type"], "members": list(v["members"])} for k, v in (groups or {}).items()}
        self.cases = dict(cases or {})

    def case_names(self):
        out = []

        def walk(it):
            if it in self.groups:
                for m in self.groups[it]["members"]:
                    walk(m)
            elif it not in out:
                out.append(it)

        for _, it in self.terms:
            walk(it)
        return out

    def case_loads(self, case):
        return list(self.cases.get(case, []))

    def describe(self):
        return " + ".join(f"{f:g}*{it}" for f, it in self.terms)


def is_group_combo(loads):
    return isinstance(loads, GroupLoads)


def envelope(gl, values):
    """(max, min) of a response for the combination; values = {case: array}."""
    def item(it):
        if it not in gl.groups:
            v = np.asarray(values[it], dtype=float)
            return v, v
        g = gl.groups[it]
        subs = [item(m) for m in g["members"]]
        if g["type"] == "Linear Add":
            return sum(s[0] for s in subs), sum(s[1] for s in subs)
        if g["type"] == "Range Add":
            return (sum(np.maximum(s[0], 0.0) for s in subs), sum(np.minimum(s[1], 0.0) for s in subs))
        return np.maximum.reduce([s[0] for s in subs]), np.minimum.reduce([s[1] for s in subs])

    hi = lo = 0.0
    for f, it in gl.terms:
        mx, mn = item(it)
        if f >= 0:
            hi, lo = hi + f * mx, lo + f * mn
        else:
            hi, lo = hi + f * mn, lo + f * mx
    return np.asarray(hi, dtype=float), np.asarray(lo, dtype=float)


def governing(hi, lo):
    """Value of larger magnitude at every point (for single-curve reports)."""
    hi, lo = np.asarray(hi, dtype=float), np.asarray(lo, dtype=float)
    return np.where(np.abs(hi) >= np.abs(lo), hi, lo)


def _envelope_choices(gl, limit=64):
    """All ways of replacing each Envelope group by one of its members.

    [None] when the combination has no Envelope group (or too many choices)."""
    env_groups = []

    def walk(it):
        if it in gl.groups:
            if gl.groups[it]["type"] == "Envelope" and it not in env_groups:
                env_groups.append(it)
            for m in gl.groups[it]["members"]:
                walk(m)

    for _, it in gl.terms:
        walk(it)
    if not env_groups:
        return [None]
    opts = [gl.groups[g]["members"] for g in env_groups]
    if int(np.prod([len(o) for o in opts])) > limit:
        return [None]
    import itertools
    return [dict(zip(env_groups, ch)) for ch in itertools.product(*opts)]


def _with_choice(gl, choice):
    if not choice:
        return gl
    groups = {k: {"type": v["type"], "members": list(v["members"])} for k, v in gl.groups.items()}
    for g, m in choice.items():
        groups[g] = {"type": "Linear Add", "members": [m]}
    return GroupLoads((), gl.terms, groups, gl.cases)


def governing_curve(gl, values, choices=None):
    """One curve of a member for drawing and reports.

    Range Add and Linear Add: at every section the value of larger magnitude
    between maximum and minimum. Envelope groups (e.g. W+ / W-, whose results
    have opposite signs): the member case that gives the largest value in this
    member, so the curve is the real diagram of that case and has no jumps.
    The peak value is the same as the peak of the envelope."""
    best, best_peak = None, -1.0
    for ch in (choices if choices is not None else _envelope_choices(gl)):
        g = governing(*envelope(_with_choice(gl, ch), values))
        peak = float(np.max(np.abs(g))) if g.size else 0.0
        if peak > best_peak * (1.0 + 1e-9) + 1e-12:
            best, best_peak = g, peak
    return best


class NVMEnvelope(dict):
    """{eid: (xg, yg, (Nmax, Nmin), (Vmax, Vmin), (Mmax, Mmin), L, c, s)} plus
    .governing = {eid: (N, V, M)}, the single curves used for drawing."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.governing = {}


# ------------------------------------------------------------------
# analysis of the basic cases (with a small cache per structural model)
# ------------------------------------------------------------------
_CACHE = OrderedDict()
_CACHE_SIZE = 24


def _model_key(parts):
    out = []
    for p in parts:
        if isinstance(p, dict):
            out.append(tuple(sorted((k, repr(v)) for k, v in p.items())))
        elif isinstance(p, (list, tuple)):
            out.append(tuple(repr(v) for v in p))
        else:
            out.append(repr(p) if not callable(p) else id(p))
    return tuple(out)


def analyse_cases(gl, build, apply_loads, run, collect, model_key=None):
    """Analyse every basic case of `gl` on the same model.

    build()            builds the model
    apply_loads(loads) applies one load case
    run()              runs the static analysis (0 = ok)
    collect(case)      reads the results needed
    Returns {case: collect(case)} or None when an analysis fails.
    """
    out = {}
    for case in gl.case_names():
        key = None
        if model_key is not None:
            key = (model_key, case, tuple(repr(L) for L in gl.cases.get(case, [])))
            if key in _CACHE:
                _CACHE.move_to_end(key)
                out[case] = _CACHE[key]
                continue
        build()
        apply_loads(gl.case_loads(case))
        if run() != 0:
            return None
        out[case] = collect(case)
        if key is not None:
            _CACHE[key] = out[case]
            while len(_CACHE) > _CACHE_SIZE * 16:
                _CACHE.popitem(last=False)
    return out


def clear_cache():
    _CACHE.clear()


# ------------------------------------------------------------------
# helpers used by the different parts of the program
# ------------------------------------------------------------------
def nvm_envelope(gl, elements, nodes, per_case_arrays):
    """{eid: (xg, yg, (Nmax, Nmin), (Vmax, Vmin), (Mmax, Mmin), L, c, s)} from per-case N/V/M arrays,
    with .governing = {eid: (N, V, M)} single curves (see governing_curve).

    per_case_arrays: {case: {eid: _nvm_arrays_for_element(...) result}}
    """
    out = NVMEnvelope()
    cases = list(per_case_arrays)
    choices = _envelope_choices(gl)
    for eid in elements:
        res = [per_case_arrays[c].get(eid) for c in cases]
        if any(r is None for r in res):
            continue
        xg, yg, _, _, _, L, c, s = res[0]
        trip, gov = [], []
        for k in (2, 3, 4):
            vals = {cs: r[k] for cs, r in zip(cases, res)}
            trip.append(envelope(gl, vals))
            gov.append(governing_curve(gl, vals, choices))
        out[eid] = (xg, yg, trip[0], trip[1], trip[2], L, c, s)
        out.governing[eid] = tuple(gov)
    return out


def story_drift_envelope(geom, nodes, per_case_U, gl):
    """Storey drifts in the format of static_analysis.compute_story_drifts, from the
    envelope of the storey drifts (each column line) over the combination."""
    from framekit.geometry import node_id_at
    nfp = geom.num_bays + 1
    nf = geom.num_floors
    cases = list(per_case_U)
    du = {c: np.zeros((nf, nfp)) for c in cases}
    ut = {c: np.zeros((nf, nfp)) for c in cases}
    for c in cases:
        U = per_case_U[c]
        for st in range(1, nf + 1):
            for ix in range(nfp):
                top = float(U[node_id_at(geom, st, ix)][0])
                bot = float(U[node_id_at(geom, st - 1, ix)][0])
                du[c][st - 1, ix] = top - bot
                ut[c][st - 1, ix] = top
    dhi, dlo = envelope(gl, du)
    uhi, ulo = envelope(gl, ut)
    drifts = []
    h = float(geom.floor_height)
    for st in range(1, nf + 1):
        pairs = []
        for ix in range(nfp):
            d = dhi[st - 1, ix] if abs(dhi[st - 1, ix]) >= abs(dlo[st - 1, ix]) else dlo[st - 1, ix]
            u = uhi[st - 1, ix] if abs(uhi[st - 1, ix]) >= abs(ulo[st - 1, ix]) else ulo[st - 1, ix]
            pairs.append({"story": st, "grid_x": ix, "node_bot": node_id_at(geom, st - 1, ix),
                          "node_top": node_id_at(geom, st, ix), "u_bot_m": u - d, "u_top_m": u,
                          "delta_u_m": d, "drift_abs_m": abs(d), "drift_ratio": abs(d) / h,
                          "cum_disp_top_m": abs(u)})
        ctrl = max(pairs, key=lambda p: p["drift_ratio"])
        drifts.append({"story": st, "max_drift_m": ctrl["drift_abs_m"], "max_drift_ratio": ctrl["drift_ratio"],
                       "max_cum_disp_m": ctrl["cum_disp_top_m"], "control_grid_x": ctrl["grid_x"],
                       "control_node_bot": ctrl["node_bot"], "control_node_top": ctrl["node_top"],
                       "all_positions": pairs})
    return drifts


def node_disps_by_case(gl, nodes, build, apply_loads, run, model_key=None):
    from framekit.static_analysis import get_node_disps
    return analyse_cases(gl, build, apply_loads, run, lambda case: get_node_disps(nodes), model_key=model_key)


def nvm_by_case(gl, elements, nodes, build, apply_loads, run, npts=60, element_ids=None, model_key=None):
    from framekit.diagrams import _nvm_arrays_for_element
    ids = list(element_ids) if element_ids is not None else list(elements)

    def collect(case):
        loads = gl.case_loads(case)
        return {eid: _nvm_arrays_for_element(eid, elements, loads, nodes, npts=npts) for eid in ids}

    return analyse_cases(gl, build, apply_loads, run, collect, model_key=model_key)
