"""Load groups: Linear Add, Range Add and Envelope of load cases (as in SAP2000).

A group combines load cases (or other groups):

  Linear Add   all members together
  Range Add    each member counted only where it makes a result larger (for the
               maximum) or smaller (for the minimum): pattern loading
  Envelope     the largest / smallest result of the members, one at a time

Range Add and Envelope are combined at result level (framekit.superposition):
every basic case is analysed once and the results are superposed, so a group is
one combination, not one combination per load pattern.

Use in the Eurocode combinations:
  Q     variable (imposed) load        e.g. CROWD = Range Add (live01..live04)
  mass  seismic mass and seismic       e.g. CROWD_SEISM = Linear Add (live01..live04)
        combination: SW + G + psi x mass group (all members loaded)
W+ and W- stay separate in the Eurocode combinations; WIND = Envelope (W+; W-)
is available as a group for your own combinations and to look at its results.
"""
from framekit.input_helpers import ask_choice_def, ask_float_def

TYPES = {"1": "Linear Add", "2": "Range Add", "3": "Envelope"}


def describe(groups, name):
    g = groups[name]
    return f"{g['type']} ({'; '.join(g['members'])})"


def _member_choice(prompt, choices, default):
    valid = {c.lower(): c for c in choices}
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip()]
        if len(parts) == 1 and parts[0].lower() == "all":
            return list(default)
        bad = [p for p in parts if p.lower() not in valid]
        if not bad and parts:
            return [valid[p.lower()] for p in parts]
        print(f"  ❌ Unknown: {', '.join(bad) or '(empty)'}. Choose among: {'; '.join(choices)}")


def _choose_role(prompt, options, default):
    if not options:
        return None
    opts = list(options) + ["none"]
    v = ask_choice_def(f"{prompt} ({' / '.join(opts)}) [{default or 'none'}]: ",
                       [o.lower() for o in opts], (default or "none").lower())
    for o in options:
        if o.lower() == v:
            return o
    return None


def ask_load_groups(cases):
    """Interactive definition of the load groups and of their roles.

    Returns None when there are no live load cases (the program then keeps
    the standard combinations with Q, W+ and W-).
    """
    basic = [k for k, v in cases.items() if v and k not in ("SW", "G")]
    live = [k for k in basic if k not in ("W+", "W-")]
    wind = [k for k in ("W+", "W-") if cases.get(k)]
    if not [k for k in live if k != "Q"]:
        return None

    groups = {"CROWD": {"type": "Range Add", "members": list(live)},
              "CROWD_SEISM": {"type": "Linear Add", "members": list(live)}}
    if wind:
        groups["WIND"] = {"type": "Envelope", "members": list(wind)}

    print("\n--- Load groups ---")
    print("Linear Add = all members together; Range Add = each member only where it is unfavourable")
    print("(pattern loading); Envelope = largest / smallest of the members. Results are superposed,")
    print("so each group is one combination, as in SAP2000.")
    print("Proposed groups:")
    for g in groups:
        print(f"  {g:<12s}: {describe(groups, g)}")
    if ask_choice_def("Keep these groups? (y/n) [y]: ", ["y", "n"], "y") == "n":
        groups = {}
        print("Define the groups (Enter at the name to stop).")
        while True:
            name = input(f"Group name{' (Enter = stop)' if groups else ''}: ").strip().upper().replace(" ", "_")
            if not name:
                if groups:
                    break
                print("  ❌ Define at least one group.")
                continue
            if name in cases or name in ("SW", "G"):
                print("  ❌ This name is a load case; choose another one.")
                continue
            t = TYPES[ask_choice_def("  Type: 1) Linear Add  2) Range Add  3) Envelope [1]: ", ["1", "2", "3"], "1")]
            choices = basic + [g for g in groups]
            mem = _member_choice(f"  Members ({'; '.join(choices)}), separated by ; [all live cases]: ",
                                 choices, live)
            groups[name] = {"type": t, "members": mem}
            print(f"  {name}: {describe(groups, name)}")

    names = list(groups)
    q_default = "CROWD" if "CROWD" in groups else names[0]
    m_default = "CROWD_SEISM" if "CROWD_SEISM" in groups else q_default
    print("\nUse of the groups in the Eurocode combinations (W+ and W- stay separate):")
    role_q = _choose_role("  Variable load Q", names, q_default)
    role_m = _choose_role("  Live load in the seismic mass", names, m_default)
    psi = ask_float_def("  Seismic mass = SW + G + psi x that group (all members loaded);  psi [0.3]: ", 0.3,
                        min_val=0.0, max_val=1.0)
    out = {"groups": groups, "Q": role_q, "mass": role_m, "psi_mass": float(psi)}
    print_load_groups(out)
    return out


def print_load_groups(lg):
    print("\n=== LOAD GROUPS ===")
    for g in lg["groups"]:
        print(f"  {g:<12s}: {describe(lg['groups'], g)}")
    print(f"  Q in the Eurocode combinations: {lg['Q'] or '-'}  (W+ and W- separate)")
    m = lg["mass"]
    print(f"  Seismic mass: SW + G + {lg['psi_mass']:g} x {m}" if m else "  Seismic mass: SW + G")
