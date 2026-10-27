import re

def ask_int(prompt, min_val=None, max_val=None):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
        except ValueError:
            print("❌ Please enter an integer.")
            continue
        if min_val is not None and v < min_val:
            print(f"❌ Must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"❌ Must be <= {max_val}.")
            continue
        return v

def ask_float(prompt, min_val=None, max_val=None, allow_zero=True):
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            v = float(s)
        except ValueError:
            print("❌ Please enter a number.")
            continue
        if not allow_zero and abs(v) < 1e-15:
            print("❌ Zero is not allowed.")
            continue
        if min_val is not None and v < min_val:
            print(f"❌ Must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"❌ Must be <= {max_val}.")
            continue
        return v

def ask_choice(prompt, choices):
    choices_l = [c.lower() for c in choices]
    while True:
        s = input(prompt).strip().lower()
        if s in choices_l:
            return s
        print(f"❌ Choose one of: {', '.join(choices)}")

def ask_yes_no(prompt):
    return ask_choice(prompt, ["y", "n"]) == "y"

def parse_id_list(spec: str, valid_ids: set):
    spec = spec.strip().lower()
    if spec == "all":
        return sorted(valid_ids)

    parts = re.split(r"[,\s]+", spec)
    out = set()
    for p in parts:
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                ia = int(a)
                ib = int(b)
            except ValueError:
                continue
            if ia > ib:
                ia, ib = ib, ia
            for k in range(ia, ib + 1):
                if k in valid_ids:
                    out.add(k)
        else:
            try:
                k = int(p)
            except ValueError:
                continue
            if k in valid_ids:
                out.add(k)
    return sorted(out)

def ask_element_ids(prompt, elements):
    valid_ids = set(elements.keys())
    while True:
        s = input(prompt).strip()
        ids = parse_id_list(s, valid_ids)
        if ids:
            return ids
        print("❌ Invalid selection. Examples: all | 5 | 2,3,10-12 | 1,3-5,9")



# ------------------------------------------------------------
# Prompts with defaults (Enter = default); they re-ask on bad input
# ------------------------------------------------------------
def ask_float_def(prompt, default=None, min_val=None, max_val=None):
    while True:
        s = input(prompt).strip().replace(",", ".")
        if not s:
            if default is not None:
                return float(default)
            print("❌ Please enter a number.")
            continue
        try:
            v = float(s)
        except ValueError:
            print("❌ Please enter a number.")
            continue
        if min_val is not None and v < min_val:
            print(f"❌ Must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"❌ Must be <= {max_val}.")
            continue
        return v


def ask_int_def(prompt, default=None, min_val=None, max_val=None):
    while True:
        s = input(prompt).strip()
        if not s:
            if default is not None:
                return int(default)
            print("❌ Please enter an integer.")
            continue
        try:
            v = int(s)
        except ValueError:
            print("❌ Please enter an integer.")
            continue
        if min_val is not None and v < min_val:
            print(f"❌ Must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"❌ Must be <= {max_val}.")
            continue
        return v


def ask_choice_def(prompt, choices, default=None):
    choices_l = [str(c).lower() for c in choices]
    while True:
        s = input(prompt).strip().lower()
        if not s and default is not None:
            return str(default).lower()
        if s in choices_l:
            return s
        print(f"❌ Choose one of: {', '.join(str(c) for c in choices)}")


def ask_existing_file(prompt):
    import os
    while True:
        p = input(prompt).strip().strip('"').strip("'")
        if p and os.path.isfile(p):
            return os.path.abspath(p)
        home = os.environ.get("ARCOPY_HOME", "")
        if p and home and os.path.isfile(os.path.join(home, p)):
            return os.path.join(home, p)
        print(f"❌ File not found: {p!r}. Enter a valid path (relative to the main.py folder or absolute).")
