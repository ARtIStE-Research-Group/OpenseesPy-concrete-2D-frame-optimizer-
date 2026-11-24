"""Number of sections (stations) along each member at which N, V and M are evaluated.

The internal forces of a member are computed at equally spaced sections,
both ends included. The same sections are used for the N/V/M diagrams, the
EC2 member checks, the seismic checks, the retrofit evaluation and the
report. Fewer sections make the checks faster; more sections locate the
maximum moment inside the span more precisely (an odd number always
includes midspan).

When no number is set (None), every routine keeps its built-in value
(41 sections for the checks and the report, 60-80 for the diagrams).
"""
from framekit.input_helpers import ask_int_def

_SECTIONS = {"beam": None, "column": None}


def set_member_sections(n_beam=None, n_column=None):
    _SECTIONS["beam"] = int(n_beam) if n_beam else None
    _SECTIONS["column"] = int(n_column) if n_column else None


def get_member_sections():
    return dict(_SECTIONS)


def npts_for(element_type, default):
    """Number of sections for a member of the given type ('beam' or 'column')."""
    n = _SECTIONS.get(str(element_type).lower())
    return int(n) if n else int(default)


def ask_member_sections():
    """Ask the number of sections for beams and for columns (Enter = built-in values)."""
    print("\n--- Sections along each member for N/V/M ---")
    print("N, V and M are evaluated at equally spaced sections along each member, both ends included.")
    print("Fewer sections = faster checks; an odd number always includes midspan.")
    print("Press Enter to keep the built-in values (41 for the checks, 60-80 for the diagrams).")
    out = []
    for kind in ("BEAM", "COLUMN"):
        n = ask_int_def(f"Number of sections along each {kind} [Enter = built-in]: ", default=0, min_val=0)
        if n == 0:
            out.append(None)
        elif n < 3:
            print("  At least 3 sections are needed (both ends and midspan): 3 is used.")
            out.append(3)
        else:
            out.append(n)
    txt = lambda n: "built-in" if n is None else f"{n} sections"
    print(f"Beams: {txt(out[0])};  columns: {txt(out[1])}")
    return out[0], out[1]
