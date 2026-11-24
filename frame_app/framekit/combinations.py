import copy

from framekit.input_helpers import ask_choice_def, ask_float_def

from framekit.constants import GAMMA_G_UNF, GAMMA_Q, PSI_Q0, PSI_Q1, PSI_Q2, PSI_W0, PSI_W2


def scale_loads(loads, factor):
    out = []
    for Ld in loads:
        Lc = copy.deepcopy(Ld)

        if Lc.load_type == "triangular":
            if Lc.w1 is not None:
                Lc.w1 *= factor
            if Lc.w2 is not None:
                Lc.w2 *= factor
        else:
            Lc.value *= factor

        out.append(Lc)
    return out


def combine_loadcases(case_dict, factors_dict):
    combo = []
    for k, fac in factors_dict.items():
        if abs(fac) < 1e-15:
            continue
        combo += scale_loads(case_dict.get(k, []), fac)
    return combo


def build_eurocode_combinations(case_dict, load_groups=None):
    if load_groups:
        return build_group_combinations(case_dict, load_groups)
    combos = {}

    # --------------------------------------------------------
    # ULS
    # --------------------------------------------------------
    combos["ULS_Q_leading_W+"] = {
        "factors": {
            "SW": GAMMA_G_UNF,
            "G": GAMMA_G_UNF,
            "Q": GAMMA_Q,
            "W+": GAMMA_Q * PSI_W0
        },
        "note": "ULS (Q leading, W+ accompanying)"
    }

    combos["ULS_Q_leading_W-"] = {
        "factors": {
            "SW": GAMMA_G_UNF,
            "G": GAMMA_G_UNF,
            "Q": GAMMA_Q,
            "W-": GAMMA_Q * PSI_W0
        },
        "note": "ULS (Q leading, W- accompanying)"
    }

    combos["ULS_W+_leading"] = {
        "factors": {
            "SW": GAMMA_G_UNF,
            "G": GAMMA_G_UNF,
            "W+": GAMMA_Q,
            "Q": GAMMA_Q * PSI_Q0
        },
        "note": "ULS (W+ leading)"
    }

    combos["ULS_W-_leading"] = {
        "factors": {
            "SW": GAMMA_G_UNF,
            "G": GAMMA_G_UNF,
            "W-": GAMMA_Q,
            "Q": GAMMA_Q * PSI_Q0
        },
        "note": "ULS (W- leading)"
    }

    # --------------------------------------------------------
    # SLS
    # --------------------------------------------------------
    combos["SLS_char_W+"] = {
        "factors": {
            "SW": 1.0,
            "G": 1.0,
            "Q": 1.0,
            "W+": PSI_W0
        },
        "note": "SLS characteristic with W+"
    }

    combos["SLS_char_W-"] = {
        "factors": {
            "SW": 1.0,
            "G": 1.0,
            "Q": 1.0,
            "W-": PSI_W0
        },
        "note": "SLS characteristic with W-"
    }

    combos["SLS_frequent_W+"] = {
        "factors": {
            "SW": 1.0,
            "G": 1.0,
            "Q": PSI_Q1,
            "W+": PSI_W2
        },
        "note": "SLS frequent with W+"
    }

    combos["SLS_frequent_W-"] = {
        "factors": {
            "SW": 1.0,
            "G": 1.0,
            "Q": PSI_Q1,
            "W-": PSI_W2
        },
        "note": "SLS frequent with W-"
    }

    combos["SLS_quasi_perm"] = {
        "factors": {
            "SW": 1.0,
            "G": 1.0,
            "Q": PSI_Q2
        },
        "note": "SLS quasi-permanent"
    }

    out = {}
    for name, data in combos.items():
        out[name] = {
            "loads": combine_loadcases(case_dict, data["factors"]),
            "note": data["note"],
            "factors": data["factors"]
        }

    return out


# ============================================================
# EUROCODE COMBINATIONS WITH LOAD GROUPS (result level, as in SAP2000)
# The Eurocode set keeps its 9 combinations: the variable load Q is the
# group chosen for Q (e.g. CROWD = Range Add of the live cases), W+ and W-
# stay separate. The groups themselves are added as combinations (e.g.
# CROWD, CROWD_SEISM, WIND) to look at their results. Quasi-permanent =
# seismic mass: SW + G + psi x (members of the mass group, all loaded).
# ============================================================
def _flatten_linear(item, groups, factor=1.0, out=None):
    """{case: factor} when only Linear Add groups are involved, else None."""
    out = {} if out is None else out
    if item not in groups:
        out[item] = out.get(item, 0.0) + factor
        return out
    g = groups[item]
    if g["type"] != "Linear Add":
        return None
    for m in g["members"]:
        if _flatten_linear(m, groups, factor, out) is None:
            return None
    return out


def _all_members(item, groups, factor=1.0, out=None):
    """{case: factor} with every member of the group loaded (used for the seismic mass)."""
    out = {} if out is None else out
    if item not in groups:
        out[item] = out.get(item, 0.0) + factor
        return out
    for m in groups[item]["members"]:
        _all_members(m, groups, factor, out)
    return out


def make_combo(case_dict, items, groups, note):
    """Combination from {item: factor} (cases and groups).

    Only Linear Add groups -> ordinary list of loads; a Range Add or Envelope
    group -> GroupLoads, combined at result level by superposition.
    """
    from framekit.superposition import GroupLoads
    flat = {}
    for it, f in items.items():
        sub = _flatten_linear(it, groups, f)
        if sub is None:
            flat = None
            break
        for k, v in sub.items():
            flat[k] = flat.get(k, 0.0) + v
    if flat is not None:
        flat = {k: v for k, v in flat.items() if abs(v) > 1e-15}
        return {"loads": combine_loadcases(case_dict, flat), "note": note, "factors": dict(items)}
    shown = {}
    for it, f in items.items():
        shown.update({k: v for k, v in _all_members(it, groups, f).items()})
    used = set()
    for it in items:
        used |= set(_all_members(it, groups))
    gl = GroupLoads(combine_loadcases(case_dict, shown), terms=[(f, it) for it, f in items.items()],
                    groups=groups, cases={c: case_dict.get(c, []) for c in used})
    return {"loads": gl, "note": note + "  [result-level: Range Add / Envelope]", "factors": dict(items)}


def build_group_combinations(case_dict, load_groups):
    groups = load_groups["groups"]
    q = load_groups.get("Q")
    psi = float(load_groups.get("psi_mass", PSI_Q2))
    has_w = {w: bool(case_dict.get(w)) for w in ("W+", "W-")}

    def items(sw_g, q_f, w=None, w_f=0.0):
        d = {"SW": sw_g, "G": sw_g}
        if q and abs(q_f) > 1e-15:
            d[q] = q_f
        if w and abs(w_f) > 1e-15:
            d[w] = w_f
        return d

    T = []
    for w in ("W+", "W-"):
        T.append((f"ULS_Q_leading_{w}", f"ULS (Q leading, {w} accompanying)", items(GAMMA_G_UNF, GAMMA_Q, w, GAMMA_Q * PSI_W0)))
    for w in ("W+", "W-"):
        T.append((f"ULS_{w}_leading", f"ULS ({w} leading)", items(GAMMA_G_UNF, GAMMA_Q * PSI_Q0, w, GAMMA_Q)))
    for w in ("W+", "W-"):
        T.append((f"SLS_char_{w}", f"SLS characteristic with {w}", items(1.0, 1.0, w, PSI_W0)))
    for w in ("W+", "W-"):
        T.append((f"SLS_frequent_{w}", f"SLS frequent with {w}", items(1.0, PSI_Q1, w, PSI_W2)))
    mass = load_groups.get("mass")
    qp = {"SW": 1.0, "G": 1.0}
    if mass and psi > 0:
        for k, v in _all_members(mass, groups, psi).items():
            qp[k] = qp.get(k, 0.0) + v
    out = {}
    for name, note, it in T:
        if any(k in ("W+", "W-") and not has_w[k] for k in it):
            it = {k: v for k, v in it.items() if k not in ("W+", "W-") or has_w[k]}
        out[name] = make_combo(case_dict, it, groups, note)
    out["SLS_quasi_perm"] = {"loads": combine_loadcases(case_dict, qp),
                             "note": f"SLS quasi-permanent = seismic mass: SW + G + {psi:g} x {mass or '-'}",
                             "factors": qp}
    for g in groups:                       # the groups themselves, to look at their results
        out[g] = make_combo(case_dict, {g: 1.0}, groups, f"Load group {g}: {groups[g]['type']}")
    return out


def _combo_str(factors: dict):
    first = ["SW", "G", "Q"]
    last = ["W+", "W-", "E"]
    keys = [k for k in first if k in factors] + \
           [k for k in factors if k not in first and k not in last] + [k for k in last if k in factors]
    parts = [f"{factors[k]:.3g}*{k}" for k in keys if abs(factors[k]) > 1e-15]
    return " + ".join(parts) if parts else "(empty)"


def print_combinations_detailed(combos: dict):
    names = list(combos.keys())

    print("\n=== Load combinations (details) ===")
    if len(names) > 20:      # many load patterns: one line per combination
        for i, cname in enumerate(names, start=1):
            print(f"{i:3d}) {cname:<44s} {_combo_str(combos[cname]['factors'])}")
        return names
    for i, cname in enumerate(names, start=1):
        f = combos[cname]["factors"]
        print(f"{i:2d}) {cname}")
        print(f"    {combos[cname]['note']}")
        print(f"    Combo: {_combo_str(f)}")
        print(f"    Total loads generated: {len(combos[cname]['loads'])}")

    return names

def ask_user_combinations(case_dict, combos, load_groups=None):
    """Optional load combinations defined by the user, added after the automatic ones.

    The name starts with ULS_ or SLS_, so a ULS combination is used everywhere the
    automatic ULS combinations are (EC2 checks, deformed shapes, retrofit, report).
    Factors can be given to load cases and to load groups; Range Add and Envelope
    groups are combined at result level (one combination, as in SAP2000).
    """
    groups = (load_groups or {}).get("groups", {})
    in_groups = {m for g in groups.values() for m in g["members"]}
    order = [k for k in ["SW", "G"] if case_dict.get(k)] + list(groups) + \
            [k for k, v in case_dict.items() if v and k not in ("SW", "G") and k not in in_groups]
    added = []
    while ask_choice_def("\nAdd another load combination of your own? (y/n) [n]: ", ["y", "n"], "n") == "y":
        kind = ask_choice_def("  Limit state (ULS/SLS) [ULS]: ", ["uls", "sls"], "uls").upper()
        k = 1
        while f"{kind}_user_{k}" in combos:
            k += 1
        default_name = f"{kind}_user_{k}"
        raw = input(f"  Name [{default_name}]: ").strip()
        name = "".join(ch if (ch.isalnum() or ch in "+-_.") else "_" for ch in raw) or default_name
        if not name.upper().startswith(kind + "_"):
            name = f"{kind}_{name}"
        while name in combos:
            name += "_b"
        print("  Factors (Enter = 0):")
        items = {}
        for item in order:
            lab = f"{item} = {groups[item]['type']}" if item in groups else item
            f = ask_float_def(f"    factor for {lab} [0]: ", 0.0)
            if abs(f) > 1e-15:
                items[item] = f
        if not items:
            print("  All factors are 0: combination not added.")
            continue
        combos[name] = make_combo(case_dict, items, groups, f"{kind} user-defined")
        added.append(name)
        print(f"  ✔ Added {name}: {_combo_str(items)}")
    return combos, added
