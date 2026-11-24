# Verification against SAP2000

This folder holds the comparison of ARCOPY with SAP2000 v24.2.0 reported in the paper:
the two-storey, two-bay portal (Section 3) and the six frames before retrofit (Section 4.1).

The SAP2000 models are not drawn by hand. Python scripts build them through the SAP2000 API
(OAPI, package `comtypes`) from the data of the ARCOPY runs, run them and export the result
tables to CSV. The comparison scripts then pair every SAP2000 value with the ARCOPY value.

## Results

| Model | Static results | Periods | Mode shapes (MAC) | Response spectrum |
|---|---|---|---|---|
| Portal, 10 members | 9,282 values, all equal to the last digit printed by SAP2000 | equal (0.819299 s and 0.283436 s) | 1.000000 | within 0.002 % (base shear 167.647 and 167.648 kN) |
| Six frames, 44 to 120 members | 228,564 values, all equal to the last printed digit | within 0.001 % | at least 0.999999 | 2,994 values: within 0.007 % for values of at least 5 kN, kNm or mm; up to 0.11 % for small printed values (rounding) |

"Equal to the last printed digit" means that the difference is not larger than half of the last
digit that SAP2000 prints (0.001 kN, 0.0001 kNm, 0.001 mm, 0.001 mrad). The differences have a
standard deviation of 0.28 to 0.29 digits, which is the value of a pure rounding error (0.289).

The portal was run with two SAP2000 seismic models:

- `same_mass`: the mass of ARCOPY (the mass of each floor in X only, split equally among the
  joints of the floor). This is the model of the table above.
- `sap_mass`: the mass that SAP2000 computes itself from the elements and the loads (tributary
  mass at each joint, in X and Z). Periods differ by 0.02 %, the base shear by 0.10 % and the
  axial forces of the beams change (2.6 to 8.2 kN and 6.9 to 1.9 kN), because they depend on how
  the floor mass is split among the joints. The 0.10 % comes from g (9.81 in ARCOPY, 9.80665 in
  SAP2000) and from the linear interpolation of the spectrum given with 85 points.

## Files

| File or folder | Content |
|---|---|
| `sap_build_test_portal.py` | builds and runs the portal in SAP2000 (static model and seismic model with the SAP2000 mass) |
| `sap_portal_same_mass.py` | seismic model of the portal with the ARCOPY mass; needs `sap_build_test_portal.py` in the same folder |
| `sap_build_six_frames.py` | builds and runs the six frames (two models per frame) from `six_frames_data.json` |
| `make_sap_data.py` | writes `six_frames_data.json` from the ARCOPY baseline runs (`<frame>_base`) |
| `test_portal/` | SAP2000 output for the portal: the three `.sdb` models, `results_*` (result tables), `model_check_*` (input tables), logs of every API call |
| `six_frames/<frame>/` | SAP2000 output for each frame: `<frame>_static.sdb`, `<frame>_seismic.sdb`, `results_static`, `results_seismic`, `model_check`; `six_frames/build_log.txt` |
| `portal_comparison/` | `compare_portal_sap.py`, `make_portal_figures.py`, paired values (`pairs_*.csv`), `statistics.csv`, `summary.json`, figures |
| `six_frames_comparison/` | `compare_frames.py`, `make_six_figures.py`, paired values (`pairs_static.zip`, `pairs_rsa.csv`), `modal.csv`, `drifts.csv`, `scalars.csv`, `statistics.csv`, `summary.json`, figure |

In the `pairs_*` files each row is one result: load case or combination, quantity, member and
station (or joint), ARCOPY value, SAP2000 value, difference, and the difference in units of the
last digit printed by SAP2000. `statistics.csv` gives the bias, the standard deviation, the
largest difference and the Bland–Altman limits of agreement.

## How the SAP2000 models are defined

- Plane frame in the X–Z plane, fixed bases, 11 output stations per member.
- Elastic frame elements without shear deformation (shear-area modifiers 0) and without rigid end zones.
- Section stiffness of ARCOPY (concrete plus top and bottom bars) through the property modifiers on A and I33.
- Static model with the full stiffness; seismic model with I33 × 0.5 (EN 1998-1, 4.3.1(7)).
- Seismic mass as in ARCOPY: joint masses in X only (the sections get a mass modifier of 0,
  because SAP2000 takes the joint masses only together with the element self mass).
- Design spectrum given as a user function; modal combination CQC with 5 % damping.

Sign conventions (ARCOPY ↔ SAP2000): beams N = P, V = −V2, M = M3; columns N = P, V = V2,
M = −M3; joints ux = U1, uy = U3, rz = −R2; supports Rx = F1, Ry = F3, Mz = −M2.

## How to repeat the comparison

The comparison scripts need only Python (no SAP2000), because the SAP2000 tables are in this folder.
From the repository root:

```
python main.py < examples/test_rc_portal/inputs.txt
python verification/sap2000/portal_comparison/compare_portal_sap.py
python verification/sap2000/portal_comparison/make_portal_figures.py

python examples/six_frames/run_all.py F4-5B_base F6-5B_base F8-5B_base F4-7B_base F6-7B_base F8-7B_base
python verification/sap2000/six_frames_comparison/compare_frames.py
python verification/sap2000/six_frames_comparison/make_six_figures.py
```

To build the SAP2000 models again you need Windows, SAP2000 and `pip install comtypes`.
Run `sap_build_test_portal.py`, `sap_portal_same_mass.py` and `sap_build_six_frames.py`
(for example from Spyder). SAP2000 opens by itself, builds and runs the models, writes the
tables and closes. If Python reports "Invalid class string", run `RegisterSAP2000.exe` from the
SAP2000 program folder once as administrator. The models here were made with SAP2000 v24.2.0;
other versions can give small differences in the last printed digits.
