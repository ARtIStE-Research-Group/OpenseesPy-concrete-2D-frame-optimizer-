import copy

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


def build_eurocode_combinations(case_dict):
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


def _combo_str(factors: dict):
    order = ["SW", "G", "Q", "W+", "W-"]
    parts = []

    for k in order:
        if k in factors and abs(factors[k]) > 1e-15:
            parts.append(f"{factors[k]:.3g}*{k}")

    return " + ".join(parts) if parts else "(empty)"


def print_combinations_detailed(combos: dict):
    names = list(combos.keys())

    print("\n=== Load combinations (details) ===")
    for i, cname in enumerate(names, start=1):
        f = combos[cname]["factors"]
        print(f"{i:2d}) {cname}")
        print(f"    {combos[cname]['note']}")
        print(f"    Combo: {_combo_str(f)}")
        print(f"    Total loads generated: {len(combos[cname]['loads'])}")

    return names