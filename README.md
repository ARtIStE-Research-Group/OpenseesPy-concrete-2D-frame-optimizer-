# ARCOPY – Seismic Assessment and Retrofit Optimization of RC Frames with OpenSeesPy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22812753.svg)](https://doi.org/10.5281/zenodo.22812753)

OpenSeesPy-based framework for the earthquake assessment of existing reinforced concrete (RC) frames and the optimization of their retrofit: Eurocode checks, drift and member performance, RC shear-wall placement, concrete jacketing, cost evaluation and automated reporting.

The framework integrates structural modelling, Eurocode-based assessment procedures, global and local retrofit strategies, comparative cost estimation, and automated reporting within a single workflow.

The current retrofit strategies include:

- **RC shear walls** as a global intervention for controlling lateral deformation and improving the lateral-force-resisting system.
- **Concrete jacketing** as a local intervention for strengthening deficient beams and columns.

The software is intended primarily for **research, preliminary retrofit screening, comparative design, and decision support**.

User manual: [Manual_Polito_Work_Station_beta_03_EN.pdf](Manual_Polito_Work_Station_beta_03_EN.pdf)

---

## What's new in v1.3.0

- **Verification against SAP2000** ([verification/sap2000](verification/sap2000)): a two-storey portal and six frames with up to 120 members, with the SAP2000 models built by script from the ARCOPY data. About 238,000 static results are equal to the last digit printed by SAP2000; periods agree within 0.001 % and response-spectrum results within 0.01 %, apart from the rounding of small printed values.
- **Examples of the paper** ([examples](examples)): the two-storey portal and the six frames (4, 6 and 8 storeys; 5 and 7 bays), with input files, a script that runs them and reference results.
- **Up to three wall segments per layout** in the retrofit search (default two), with one more physics-guided first trial of the Bayesian search. The eight-storey example frames need it.
- **NTC 2008 response spectrum** as an alternative to EN 1998-1, and a **gross-section stiffness option** (b·h instead of concrete plus bars).
- **Live load cases and load groups** (Linear Add, Range Add, Envelope), combined at result level; user-defined combinations.
- **Wind per storey** with an optional leeward part; N/V/M diagrams with tables of end and extreme values; number of sections along the members chosen by the user.
- **Tests**: 33 checks against hand calculations and 56 checks of the portal against an independent direct-stiffness calculation.

Version 1.2.0 introduced the Bayesian shear-wall search (see [docs/smart_search.md](docs/smart_search.md)), the EC8 seismic action inside the retrofit loop, the iterative concrete jacketing with detailing limits, the step checkpoints and the Word + Excel retrofit report.

Full list: [CHANGELOG.md](CHANGELOG.md).

---

## Main Features

### Structural modelling

The program interactively generates the structural model and allows the user to define:

- number of storeys and bays;
- structural geometry;
- RC beam and column sections;
- material properties;
- existing shear walls;
- section stiffness: transformed section (concrete plus bars) or gross concrete section;
- permanent, variable, wind, and self-weight load cases;
- live load cases on chosen beams, and load groups (Linear Add, Range Add, Envelope);
- wind as one distributed load per storey (or triangular), with an optional leeward part.

The finite-element model is assembled and analysed using **OpenSeesPy**.

### Static analysis

The program generates the load combinations implemented in the project and performs linear static analysis for each combination.

Available outputs include:

- nodal displacements;
- inter-storey drifts;
- deformed shapes;
- axial-force diagrams (N);
- shear-force diagrams (V);
- bending-moment diagrams (M).

Static drift results and member-response data can also be exported for subsequent processing.

### RC member verification

Reinforced-concrete member verification is performed for the implemented ULS combinations using the EC2 assessment modules included in the project.

The workflow evaluates member utilization and identifies critical beams and columns based on quantities such as:

- axial force;
- shear force;
- bending moment;
- axial-flexural utilization;
- shear utilization;
- governing member utilization.

Critical-member results can be visualized directly on the structural model.

---

## Dynamic Analysis

The framework also provides optional linear dynamic analysis capabilities.

These include:

- eigenvalue analysis;
- modal periods and mode shapes;
- modal participation assessment;
- response-spectrum analysis with the EN 1998-1 or the NTC 2008 design spectrum;
- directional response-spectrum combination;
- EC8 drift checks;
- linear time-history analysis using multiple records.

Dynamic analyses are optional and can be activated interactively during execution.

---

## Performance Assessment

Static and dynamic results can be assembled into a common performance summary.

The framework distinguishes between:

### Global deficiencies

Primarily represented by excessive storey-drift demand.

### Local deficiencies

Represented by inadequate member capacity or utilization ratios exceeding the prescribed limit.

This separation is used to determine whether a global intervention, a local intervention, or a combination of both is required.

---

# Retrofit Optimization

The retrofit module evaluates complete retrofit strategies rather than optimizing shear walls and concrete jacketing independently. A candidate strategy is a shear-wall layout (bays, storeys, thickness) together with the concrete jacketing that its remaining deficient members need, and it is judged by its total cost.

The optimization procedure follows the general sequence:

1. assess the original structure;
2. identify storey- and member-level deficiencies;
3. generate candidate shear-wall layouts;
4. analyse the structural response of each candidate;
5. identify remaining member deficiencies;
6. design concrete jacketing where required;
7. reanalyse the complete retrofitted structure;
8. evaluate constructability and detailing constraints;
9. estimate the total retrofit cost;
10. rank the complete retrofit alternatives.

## Shear-wall search methods

At the retrofit step the user chooses how wall layouts are proposed:

1. **Screening**: all layouts are ranked with a quick drift check, then the best N go through the complete evaluation (steps 4–9).
2. **Smart (Bayesian optimization)**: a Tree-structured Parzen Estimator (Optuna) proposes each new layout from the costs and feasibility of all complete strategies analysed so far. The budget is the number of complete analyses. Details in [docs/smart_search.md](docs/smart_search.md).

## RC Shear-Wall Optimization

Candidate shear-wall configurations can vary according to:

- wall location;
- number of wall segments;
- vertical extent;
- wall thickness;
- layout configuration.

The current framework supports a hybrid wall-layout search and evaluates alternative wall thicknesses.

The user sets the largest number of wall segments per layout (1 to 3, default 2). With three segments the number of layouts grows quickly (for an 8-storey, 5-bay frame from about 10,000 to about 206,000), so the smart search is then the practical choice.

To control computational cost, the user can specify the maximum number of screened shear-wall candidates (screening) or the number of complete analyses (smart search).

This allows both relatively fast engineering studies and larger research-oriented optimization runs.

## Concrete Jacketing

Concrete jacketing is applied to members that remain deficient after the selected shear-wall configuration is introduced.

The jacketing procedure considers variables including:

- jacket thickness;
- final section dimensions;
- longitudinal reinforcement;
- transverse reinforcement;
- stirrup spacing;
- reinforcement clear-spacing requirements;
- geometric limits on section enlargement.

The objective is to avoid mathematically feasible but physically impractical retrofit solutions.

If a member cannot satisfy the required verification and detailing constraints within the allowed jacketing limits, the solution is flagged rather than silently accepted.

## Global Retrofit Optimization

The optimization evaluates the **complete intervention cost** with possible shear walls and concrete jacketing candidates.

Feasible strategies must satisfy the prescribed structural and detailing constraints.

Among feasible alternatives, the framework prioritizes lower total retrofit cost.

When no fully feasible solution exists within the investigated design space, penalty terms are used to distinguish alternatives according to factors such as:

- drift exceedance;
- remaining member failures;
- constructability;
- jacketing/detailing limitations.

The optimization therefore acts as a **decision-support procedure**, rather than relying on a single unconstrained scalar performance score.

---

# Cost Model

The retrofit cost model is intended for **comparative preliminary assessment** rather than contractor-level cost estimation.

Cost components can include:

### Shear walls

- concrete;
- reinforcement;
- formwork;
- labour;
- indirect costs;
- foundation/connection allowances;
- constructability allowances;
- duration-related site overhead.

### Concrete jacketing

- additional concrete;
- reinforcement;
- formwork;
- labour;
- indirect costs;
- duration-related site overhead.

The user may use the default cost assumptions or provide project-specific unit costs before optimization.

This allows the same structural alternatives to be investigated under different regional or project-specific economic conditions.

---

# Installation

Python 3.10 or newer. Tested on Windows (Anaconda + Spyder, Python 3.10) and Linux (Python 3.13).

With conda (recommended):

```bash
conda env create -f environment.yml
conda activate arcopy
```

Or with pip in an existing environment:

```bash
pip install -r requirements.txt
```

`optuna` is needed only for the smart (Bayesian) wall search; the screening search works without it. In a Spyder console you can install it with `%pip install optuna` and then restart the kernel.

OpenSeesPy is installed from PyPI and is distributed under its own licence (see *License* below).

# Running

```bash
python main.py
```

or open `main.py` in Spyder and press Run. The program asks for the model, loads and analysis options step by step.

- Every run has a name; its results go to a folder with that name next to `main.py`.
- After each of the 12 steps the state is saved in `<run>/_checkpoints`. Starting `main.py` again with the same run name lets you continue, or restart from any earlier step without re-entering the model.

# Examples

Each example has the answers to all prompts in a text file, so it runs without typing, and its reference results.

| Folder | Content |
|---|---|
| [examples/test_rc_portal](examples/test_rc_portal) | two-storey, two-bay portal with live load cases and load groups; the verification example (paper, Section 3) |
| [examples/six_frames](examples/six_frames) | the six frames of the paper (Section 4): existing frame, screening and Bayesian search |
| [examples/four_storey_frame](examples/four_storey_frame) | four-storey, five-bay frame of v1.2.0 |

```bash
python main.py < examples/test_rc_portal/inputs.txt
python main.py < examples/six_frames/inputs/F4-5B_bayes.txt
python examples/six_frames/run_all.py
```

In Windows PowerShell use `Get-Content <file> | python main.py`. To run without figure windows set the Matplotlib backend first (`set MPLBACKEND=Agg` in cmd); `run_all.py` does it by itself.

# Verification

```bash
python tests/run_verification.py
```

compares framekit results with closed-form hand calculations (self-weight and seismic masses, N/V/M diagrams, wind loads, EN 1998-1 and NTC 2008 spectra, CQC, behaviour factor and confidence factor, lateral force method, cracked stiffness, gross-section option, jacket detailing) and prints `ALL CHECKS PASSED  (33 checks)`.

```bash
python tests/verify_test_rc_portal.py
```

runs the portal example and checks it against an independent direct-stiffness calculation written with NumPy (56 checks, about one minute).

The comparison with SAP2000 (models, result tables, scripts and statistics) is in [verification/sap2000](verification/sap2000).

---

# Outputs

Depending on the selected analyses, the program writes to the run folder files such as:

```text
story_drifts.csv
ec2_worst_utilization.xlsx

modal_periods.csv
modal_participation.csv

response_spectrum_drifts.csv
response_spectrum_results.xlsx
ec8_drift_check_single_direction.xlsx
ec8_seismic_storey_checks.xlsx
ec8_seismic_member_checks.xlsx

time_history_batch_summary.csv
time_history_batch_peak_drifts.csv
time_history_batch_results.xlsx

performance_summary.xlsx

retrofit_global_optimization.csv
retrofit_top5_alternatives.csv
smart_search_history.csv                 (smart search: every trial, cost, feasibility, best so far)

retrofit_jacketing_member_report.csv
ec2_worst_utilization_after_retrofit.xlsx

global_retrofit_optimization_dataset_later_ML.csv
global_retrofit_optimization_dataset_later_ML.xlsx

report/retrofit_report.docx              (Word retrofit report)
report/retrofit_report_data.xlsx
report/figures/
```

# Repository structure

```text
main.py                       interactive workflow (12 steps)
frame_app/framekit/           modelling, analysis, Eurocode checks, retrofit search, reporting
records/                      ground-motion records for time-history analysis
redraw_nvm.py                 redraws the N/V/M diagrams of a saved run
tests/                        verification against hand calculations and of the portal example
examples/                     input files and reference results (portal, six frames, four-storey frame)
verification/sap2000/         comparison with SAP2000: scripts, models, result tables, statistics
docs/                         smart-search description
```

---

# License

The source code of ARCOPY is released under the **MIT License** (see [LICENSE](LICENSE)).

The user manual and the documentation are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).

The logos of Politecnico di Torino and of the ARTISTE Research Group are trademarks of their owners and are not covered by these licences.

ARCOPY uses [OpenSeesPy](https://openseespydoc.readthedocs.io/), which is not included in this repository: it is installed separately with `pip` and is subject to its own licence terms, which users must respect.

See also [Disclaimer](Disclaimer) and [Scope and Limitations](Scope%20and%20Limitations).

# Citation

If you use ARCOPY in research or professional work, please cite the software release (see [CITATION.cff](CITATION.cff) and [Citation](Citation)):

> Hajiha, A., Marano, G. C., Bertagnoli, G., Sardone, L., Datta, G. (2026). ARCOPY: seismic assessment and cost-based retrofit optimization of RC frames with OpenSeesPy (v1.3.0). ARTISTE Research Group, Politecnico di Torino. Zenodo. https://doi.org/10.5281/zenodo.22812753

This DOI covers all versions and resolves to the latest release. Each release also has its own DOI on Zenodo (v1.3.0: https://doi.org/10.5281/zenodo.24494048; v1.2.0: https://doi.org/10.5281/zenodo.23686655; v1.0.0: https://doi.org/10.5281/zenodo.22812754).

# Authors

Arman Hajiha and Giuseppe Carlo Marano, with Gabriele Bertagnoli (supervision), Laura Sardone and Gaurav Datta (research group) – ARTISTE Research Group, Department of Structural, Building and Geotechnical Engineering (DISEG), Politecnico di Torino.

Questions and bug reports: please open an issue on GitHub.
