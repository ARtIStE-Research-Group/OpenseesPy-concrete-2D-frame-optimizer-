def _mm_to_m(x_mm: float) -> float:
    return x_mm / 1000.0

def _MPa_to_Pa(x_mpa: float) -> float:
    return x_mpa * 1e6

def _kN_to_N(x_kN: float) -> float:
    return x_kN * 1000.0

def _kNpm_to_Npm(x_kNpm: float) -> float:
    return x_kNpm * 1000.0

def _section_gross_area_m2(sec) -> float:
    return _mm_to_m(sec.b) * _mm_to_m(sec.h)

