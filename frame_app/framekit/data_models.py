from dataclasses import dataclass

@dataclass
class Geometry:
    num_floors: int
    floor_height: float
    num_bays: int
    bay_lengths: list

@dataclass
class Material:
    name: str
    E: float
    density: float

@dataclass
class Load:
    load_type: str
    value: float
    element: int | None = None
    node: int | None = None
    direction: str = 'vertical'
    w1: float | None = None
    w2: float | None = None

@dataclass
class FiberSection:
    b: float
    h: float
    cover: float
    top_bars: int
    top_bar_d: float
    bottom_bars: int
    bottom_bar_d: float

@dataclass
class ShearWall:
    bay_left: int
    bay_right: int
    floor_bot: int
    floor_top: int
    tw_mm: float | None = None   # own thickness; None = use the model default

@dataclass
class EC2MaterialStrengths:
    fck: float
    fyk: float
    gamma_c: float
    gamma_s: float
    alpha_cc: float
