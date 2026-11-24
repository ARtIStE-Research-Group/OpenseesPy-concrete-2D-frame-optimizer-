# Changelog

## v1.3.0 – 2026-11-24

This is the version used in the paper "ARCOPY: seismic assessment and cost-based retrofit optimization of RC frames with OpenSeesPy" (verification against SAP2000 and six example frames).

Verification against SAP2000 (`verification/sap2000`, new)
- Scripts that build the SAP2000 models from the ARCOPY data through the SAP2000 API (no number typed by hand), the SAP2000 v24.2.0 models and result tables, the comparison scripts and the paired values with their statistics.
- Two-storey, two-bay portal: 9,282 static results equal to the last digit printed by SAP2000; with the same mass model, equal periods, MAC = 1.000000 and response-spectrum results within 0.002 %.
- Six frames with 4 to 8 storeys and 5 or 7 bays: 228,564 static results equal to the last printed digit; periods within 0.001 %; response-spectrum results within 0.007 % (values of at least 5 kN, kNm or mm).

Examples (`examples/`)
- `test_rc_portal` (new): the two-storey portal, with live load cases and load groups; `tests/verify_test_rc_portal.py` checks it against an independent direct-stiffness calculation (56 checks).
- `six_frames` (new): input files of the six frames (existing frame, screening, Bayesian search), a script that runs them, and the reference results.
- `four_storey_frame`: the input file answers the new questions; the results are those of v1.2.0.

Retrofit search (`main.py`, `frame_app/framekit/smart_search.py`)
- New question "Max wall segments per layout" (1 to 3, default 2), asked before the search method, for both methods. Three segments are needed by tall frames where two walls are not enough; the number of layouts grows quickly (for an 8-storey, 5-bay frame from about 10,000 to about 206,000), so with three segments the Bayesian search is the practical choice.
- Physics-guided first trials of the Bayesian search: with three segments, one more trial with full-height walls in the middle bay and its two neighbours, with the thinnest wall.
- Input files written for v1.2.0 need one more answer (an empty line keeps the default of 2) before the search method.

Seismic action and section stiffness
- NTC 2008 spectrum (`response_spectrum.py`, `ec8_seismic.py`): the response-spectrum input now starts with "Spectrum: 1) EN 1998-1  2) NTC 2008". For NTC 2008 the site parameters ag/g, F0 and Tc*, the ground type and the topographic category are asked; S = Ss·ST, TC = Cc·Tc*, TB = TC/3, TD = 4·ag/g + 1.6 (3.2.3.2.1), and the design spectrum is the elastic one with η = 1/q and Sd ≥ 0.2·ag (3.2.3.5). Used by the RSA, the lateral force method of the retrofit, the plots and the report. Enter keeps EN 1998-1 as before.
- Section stiffness option (`opensees_model.py`, `main.py`): after the material, "1) transformed section (concrete + bars, default)  2) gross concrete section b·h", to compare with programs that ignore the bars (e.g. SAP2000 by default). Enter keeps the transformed section.

Wind loads (`frame_app/framekit/loads.py`)
- New wind profile, now the default: **one distributed load per storey**, asked for every storey from the ground storey up; the value is applied as a uniform horizontal load on the exterior column of that storey. The previous triangular profile (0 at the base, value at the top) is still available as option 2.
- New **leeward part**: after defining W+ (or W-), the program asks whether to apply at the same time, on the exterior columns of the other side, the same storey loads multiplied by a coefficient, **in the same direction** as the windward load (suction on the leeward face). The leeward part belongs to the same load case, so it enters every combination with W+ (or W-).
- In the per-storey mode, W- can reuse the W+ storey loads and coefficient, mirrored.
- A table of the windward and leeward storey loads and the total horizontal force is printed for each case.

Load combinations (`frame_app/framekit/combinations.py`, `main.py`)
- After the automatic Eurocode combinations, the program asks whether to add combinations of your own: limit state (ULS or SLS), name, and a factor for each load case. A ULS combination is used everywhere the automatic ULS combinations are (EC2 checks, deformed shapes, retrofit search, report).

Live load cases and load groups (`loads.py`, `load_groups.py` and `superposition.py` new, `combinations.py`, `ec2_checks.py`, `diagrams.py`, `report_builder.py`, `retrofit_optimizer.py`, `main.py`)
- Live (imposed) load cases: for each case a name, a uniform load [kN/m] and the beams it loads, chosen by floor and bay (several load values per case are possible).
- Load groups as in SAP2000: Linear Add, Range Add and Envelope. Proposed by default: CROWD = Range Add (live cases), CROWD_SEISM = Linear Add (live cases), WIND = Envelope (W+; W-); the groups can also be defined by hand, also as groups of groups.
- Range Add and Envelope are combined at result level, as in SAP2000: every load case is analysed once and the results are superposed (Range Add: positive and negative parts of each case; Envelope: max / min of the members). Each group is one combination. Every response (end forces, N/V/M at the sections, storey drifts) has its own maximum and minimum; the EC2 checks use the largest |N|, |V|, |M| as for the other combinations, and the column N-M interaction is checked with both the most and the least compressive N.
- The Eurocode set keeps its 9 combinations, with Q = CROWD and W+, W- separate; the groups CROWD, CROWD_SEISM and WIND are added as combinations to look at their results (12 combinations in total, plus the seismic ones and your own).
- Seismic mass and seismic combination: SW + G + psi x CROWD_SEISM (all live cases), psi given by the user (default 0.3); used by the modal analysis, the response spectrum, the lateral force method of the retrofit and the report.
- Your own combinations accept factors on the groups.
- N/V/M diagrams of a Range Add / Envelope combination show both envelope curves, as in SAP2000: the maximum as a solid line and the minimum as a dashed line; the positive part of the maximum and the negative part of the minimum are filled (largest sagging and largest hogging). The table beside the frame gives, for every member, the maximum and the minimum at the start, at the end and along the member. (A single curve taking the larger value at each section was tried first: it jumped between maximum and minimum where they have similar size, and zigzagged in members where maximum = -minimum, e.g. the middle column of a symmetric frame under CROWD.)
- Without live load cases the program keeps the previous combinations (Q, W+, W-) exactly.
- Check on the example frame with 4 live cases: the maximum and minimum moments of every member equal those of the 16 explicit load patterns (difference < 1e-9 N m); the EC2 utilizations are equal or up to 0.02 higher (non-concurrent extremes, as in SAP2000); the complete run takes 3 min.

Sections along the members (`frame_app/framekit/member_sections.py`, new)
- The program asks the number of sections along each beam and along each column at which N, V and M are evaluated (both ends included; Enter keeps the built-in values: 41 for the checks and the report, 60-80 for the diagrams). The numbers apply to the diagrams, the EC2 and seismic checks, the retrofit evaluation and the report, and are kept when a run is resumed.

N/V/M diagrams (`frame_app/framekit/diagrams.py`)
- The values are no longer written on the diagrams, where they overlapped: the member IDs are written on the frame and a table beside the frame lists, for every member, the value at the start and at the end (start = bottom of columns, left end of beams) and, for M, the maximum and the minimum along the member (for a curved diagram, inside the span). The member with the largest value is highlighted. The IDs are the element IDs of the EC2 check tables. Columns and beams are in separate tables when the frame is large.
- `plot_structure_NVM(..., value_labels=True)` keeps the previous layout with the values on the diagrams.
- `report_builder.py`: the N/V/M envelopes of the report accept different section numbers for beams and columns.

Tests
- `tests/run_verification.py`: 33 checks against hand calculations (26 in v1.2.0; new: per-storey wind with leeward part, NTC 2008 spectrum, gross-section option).
- `tests/verify_test_rc_portal.py`: 56 checks of the portal example.

Other
- `redraw_nvm.py` (new): redraws the N/V/M diagrams of a saved run from its checkpoints.

## v1.2.0 – 2026-10-27

Licence
- The source code is now released under the MIT License (`LICENSE`). Manual and documentation: CC BY 4.0.
- Added `CITATION.cff`, `requirements.txt`, `environment.yml` and `.gitignore`.
- The code is now in the repository as plain files instead of a zip archive.

New
- Bayesian shear-wall search (`framekit/smart_search.py`): Tree-structured Parzen Estimator (Optuna, multivariate, seed 42) over wall bays, storeys, number of segments and thickness. Physics-guided first trials, 10 exploratory trials, then TPE. Drift, member utilization and complete feasibility (θ, jacketing limits) are handled as constraints. Analysed layouts are cached and do not use the budget; repeated proposals are replaced by a local refinement of the best strategy. The search stops at the budget or after max(20, budget/2) proposals with nothing new to analyse. Outputs: `smart_search_history.csv`, learning curve and search map.
- EC8 seismic action in the retrofit loop (`framekit/ec8_seismic.py`): EN 1998-1 (new design) or EN 1998-3 (existing building, knowledge level and confidence factor); lateral force method for every candidate, damage-limitation drift check, θ check, strong-column/weak-beam check.
- Concrete jacketing designed as built (`framekit/jacketing.py`): new cage and bars inside the jacket, EN 1992-1-1 clear-spacing check, minimum cage, stirrup spacing; iterative increase of thickness and steel until the member passes or a limit is reached.
- Step checkpoints (`framekit/checkpoint.py`): the workflow state is saved after each of the 12 steps; a run can be resumed or restarted from any step.
- Word + Excel retrofit report (`framekit/report_builder.py`) and common figure style (`framekit/plot_style.py`).
- Verification suite `tests/run_verification.py` (hand calculations).
- Example `examples/four_storey_frame` with the full input and reference results.

Changed
- Jacketing of each failed member starts at 75 mm thickness and 150 mm stirrup spacing (or the existing spacing, if smaller).
- Learning-curve plot: the cost axis shows every analysed trial (no points hidden above the axis).
- `main.py` checks that Optuna is installed before the smart search and offers the screening search otherwise.
- Checkpoints work when Spyder re-runs `main.py` in a console that still holds the variables of a previous run.
- The version number is printed at the start of each run.

Removed
- `install_framekit_retrofit_patch.py` and `install_utilization_map_report_patch.py` (their changes are part of the modules).

## Earlier releases

- Initial public release of the OpenSeesPy 2D RC frame optimizer (code archive `Marano- Hajiha - OpenseesPy 2D RC frame Optimizer code.zip`).
