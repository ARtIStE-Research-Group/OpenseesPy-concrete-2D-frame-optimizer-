from framekit.data_models import ShearWall
from framekit.input_helpers import ask_yes_no, ask_int, ask_float


def ask_shear_walls(geom):
    walls = []
    print("\n--- Shear walls definition ---")
    print("Define a wall panel by: Story (1..num_floors) and bay (1..num_bays).\n")

    while ask_yes_no("Add a shear wall panel? (y/n): "):
        st = ask_int(f"Story number (1..{geom.num_floors}): ", 1, geom.num_floors)
        bay = ask_int(f"Bay number (1..{geom.num_bays}): ", 1, geom.num_bays)
        walls.append(
            ShearWall(
                bay_left=bay,
                bay_right=bay + 1,
                floor_bot=st - 1,
                floor_top=st
            )
        )

    return walls


def ask_shear_wall_section():
    print("\n--- Shear wall section properties ---")
    tw = ask_float("Wall thickness tw [mm] (e.g. 200): ", min_val=50.0, allow_zero=False)
    E_wall = ask_float("Wall E [MPa] (e.g. 30000 for C25/30): ", min_val=1000.0, allow_zero=False)
    density_wall = ask_float("Wall density [kg/m3] (e.g. 2500): ", min_val=100.0, allow_zero=False)
    return tw, E_wall, density_wall