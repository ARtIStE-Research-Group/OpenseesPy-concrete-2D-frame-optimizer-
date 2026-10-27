# ARCOPY – Seismic Assessment and Retrofit Optimization of RC Frames with OpenSeesPy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23686655.svg)](https://doi.org/10.5281/zenodo.23686655)

OpenSeesPy-based framework for the earthquake assessment of existing reinforced concrete (RC) frames and the optimization of their retrofit: Eurocode checks, drift and member performance, RC shear-wall placement, concrete jacketing, cost evaluation and automated reporting.

The framework integrates structural modelling, Eurocode-based assessment procedures, global and local retrofit strategies, comparative cost estimation, and automated reporting within a single workflow.

The current retrofit strategies include:

- **RC shear walls** as a global intervention for controlling lateral deformation and improving the lateral-force-resisting system.
- **Concrete jacketing** as a local intervention for strengthening deficient beams and columns.

The software is intended primarily for **research, preliminary retrofit screening, comparative design, and decision support**.

User manual: [Manual_Polito_Work_Station_beta_03_EN.pdf](Manual_Polito_Work_Station_beta_03_EN.pdf)

---

## What's new in v1.2.0

- **Bayesian shear-wall search** (Tree-structured Parzen Estimator, Optuna): each new wall layout is chosen from the results of all previous complete retrofit strategies. Physics-guided first trials, constraint handling, cache of analysed layouts and local refinement. See [docs/smart_search.md](docs/smart_search.md).
- **EC8 seismic action inside the retrofit loop**: lateral force method for every candidate, damage-limitation drift check and interstorey drift sensitivity (θ) check, EN 1998-3 knowledge level and confidence factor.
- **Iterative concrete jacketing** with detailing limits (bar clear spacing, cage, stirrup spacing, maximum jacket thickness and section enlargement).
- **Step checkpoints**: a run can be resumed from any of its 12 steps without re-entering the model.
- **Word + Excel retrofit report** with figures of the selected strategy.
- **Verification suite** (`tests/run_verification.py`): 26 checks of framekit results against closed-form hand calculations.
- Reproducible **example** of a 4-storey, 5-bay frame in `examples/four_storey_frame`.

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
- permanent, variable, wind, and self-weight load cases.

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
- EC8 response-spectrum analysis;
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

# Example

`examples/four_storey_frame` contains the 4-storey, 5-bay frame used in the paper: the answers to all prompts (`inputs.txt`) and the reference results. To reproduce it:

```bash
python main.py < examples/four_storey_frame/inputs.txt
```

See [examples/four_storey_frame/README.md](examples/four_storey_frame/README.md).

# Verification

```bash
python tests/run_verification.py
```

compares framekit results with closed-form hand calculations (self-weight and seismic masses, N/V/M diagrams, EC8 spectra and CQC, behaviour factor and confidence factor, lateral force method, cracked stiffness, jacket detailing) and prints `ALL CHECKS PASSED`.

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
tests/                        verification against hand calculations
examples/four_storey_frame/   inputs and reference results of the example frame
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

> Marano, G. C., Hajiha, A. (2026). ARCOPY: seismic assessment and cost-based retrofit optimization of RC frames with OpenSeesPy (v1.2.0). ARTISTE Research Group, Politecnico di Torino. Zenodo. https://doi.org/10.5281/zenodo.23686655

Previous version (v1.0.0): https://doi.org/10.5281/zenodo.22812754

# Authors

Giuseppe Carlo Marano and Arman Hajiha – ARTISTE Research Group, Department of Structural, Building and Geotechnical Engineering (DISEG), Politecnico di Torino.

Questions and bug reports: please open an issue on GitHub.
