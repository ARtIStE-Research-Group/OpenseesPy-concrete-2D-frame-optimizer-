https://github.com/ARtIStE-Research-Group/OpenseesPy-concrete-2D-frame-optimizer-/blob/main/Manual_Polito_Work_Station_beta_03_EN.pdf

# OpenseesPy-concrete-2D-frame-optimizer-
OpenSeesPy-based framework for earthquake assessment of existing RC structures (frames) and optimization of retrofits (Eurocode checks, drift/member performance, RC shear wall placement, concrete jacketing, cost evaluation, automated reporting)
# Seismic Assessment and Retrofit Optimization of RC Frames with OpenSeesPy

A Python/OpenSeesPy research framework for the seismic assessment and cost-based retrofit optimization of existing reinforced concrete (RC) frame structures.

The framework integrates structural modelling, Eurocode-based assessment procedures, global and local retrofit strategies, comparative cost estimation, and automated reporting within a single workflow.

The current retrofit strategies include:

- **RC shear walls** as a global intervention for controlling lateral deformation and improving the lateral-force-resisting system.
- **Concrete jacketing** as a local intervention for strengthening deficient beams and columns.

The software is intended primarily for **research, preliminary retrofit screening, comparative design, and decision support**.

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

---

### Static analysis

The program generates the load combinations implemented in the project and performs linear static analysis for each combination.

Available outputs include:

- nodal displacements;
- inter-storey drifts;
- deformed shapes;
- axial-force diagrams \(N\);
- shear-force diagrams \(V\);
- bending-moment diagrams \(M\).

Static drift results and member-response data can also be exported for subsequent processing.

---

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

The retrofit module evaluates complete retrofit strategies rather than optimizing shear walls and concrete jacketing independently.

A candidate strategy can be represented conceptually.

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

---

## RC Shear-Wall Optimization

Candidate shear-wall configurations can vary according to:

- wall location;
- number of wall segments;
- vertical extent;
- wall thickness;
- layout configuration.

The current framework supports a hybrid wall-layout search and evaluates alternative wall thicknesses.

To control computational cost, the user can specify the maximum number of screened shear-wall candidates that proceed to the more expensive complete retrofit evaluation.

This allows both relatively fast engineering studies and larger research-oriented optimization runs.

---

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

---

## Global Retrofit Optimization

The optimization evaluates the **complete intervention cost**
with possible shear walls and concrete jacketing candidates. 

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

# Optimization Outputs

Depending on the selected analyses, the program can generate files such as:

```text
story_drifts.csv

ec2_worst_utilization.xlsx

modal_periods.csv
modal_participation.csv

response_spectrum_drifts.csv
response_spectrum_results.xlsx
response_spectrum_directional_results.xlsx

time_history_batch_summary.csv
time_history_batch_peak_drifts.csv
time_history_batch_results.xlsx

performance_summary.xlsx

retrofit_global_optimization.csv
retrofit_top5_alternatives.csv

retrofit_jacketing_member_report.csv
ec2_worst_utilization_after_retrofit.xlsx

global_retrofit_optimization_dataset_later_ML.csv
global_retrofit_optimization_dataset_later_ML.xlsx
