# Changelog

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
