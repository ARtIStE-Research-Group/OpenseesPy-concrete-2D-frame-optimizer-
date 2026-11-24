# The six frames of the paper

Six RC frames with 4, 6 and 8 storeys and 5 or 7 bays (Section 4 of the paper). For each frame there
are three input files: the assessment of the existing frame (`_base`), the retrofit with the
drift-based screening (`_screen`) and the retrofit with the Bayesian search (`_bayes`). The two
eight-storey frames have one more file (`_bayes3`), the Bayesian search with up to three wall
segments per layout.

## The frames

| Frame | Storeys | Bay lengths (m) | Beams (mm) | Columns (mm) | Members | T1 (s) |
|---|---|---|---|---|---|---|
| F4-5B | 4 | 5, 5, 6, 5, 5 | 300 × 400 | 400 × 400 | 44 | 0.841 |
| F6-5B | 6 | 5, 5, 6, 5, 5 | 400 × 500 | 500 × 500 | 66 | 0.863 |
| F8-5B | 8 | 5, 5, 6, 5, 5 | 400 × 500 | 500 × 500 | 88 | 1.158 |
| F4-7B | 4 | 4, 5, 6, 5, 6, 5, 4 | 300 × 400 | 400 × 400 | 60 | 0.817 |
| F6-7B | 6 | 4, 5, 6, 5, 6, 5, 4 | 400 × 500 | 500 × 500 | 90 | 0.837 |
| F8-7B | 8 | 4, 5, 6, 5, 6, 5, 4 | 400 × 500 | 500 × 500 | 120 | 1.124 |

- Storey height 3.0 m, fixed bases; C25/30 concrete (E = 31 GPa, 2500 kg/m³), B450C steel.
- Existing building, knowledge level KL2 (confidence factor 1.20).
- Reinforcement, assumed as the existing detailing: columns 12Ø16 with Ø8 ties at 100 mm; beams
  3Ø16 at the top and at the bottom with Ø8 stirrups at 150 mm.
- Loads: self-weight; G = 23 kN/m on every beam; Q = 2.5 kN/m on the middle bay of the first floor;
  wind on the windward column line, 1.25 kN/m in the first storey and 2.5 kN/m more in each storey
  above, in both directions.
- Seismic action: NTC 2008 spectrum, ag = 0.193 g, F0 = 2.444, Tc* = 0.333 s, ground type B,
  topographic category T1, 5 % damping, q = 1.5; flexural stiffness × 0.5; α = 0.005, ν = 0.5.
- Retrofit: wall thickness 150 to 300 mm, jacket thickness up to 200 mm, ratio of jacketed to
  original section dimension up to 1.5, default unit costs; screening with N = 40 layouts, Bayesian
  search with a budget of 40 complete analyses (seed 42).

T1 is the first period of the existing frame. `make_inputs.py` holds these data and writes the files
of `inputs/`.

## Run

From the repository root:

```
python main.py < examples/six_frames/inputs/F4-5B_bayes.txt       one run (cmd, Anaconda Prompt, Linux, macOS)
python examples/six_frames/run_all.py F4-5B_bayes                 the same, without figure windows
python examples/six_frames/run_all.py                             all 20 runs
python examples/six_frames/run_all.py --jobs 2                    two runs at the same time
```

In Windows PowerShell: `Get-Content examples\six_frames\inputs\F4-5B_bayes.txt | python main.py`.

Each run writes a folder with its name next to `main.py`. Delete that folder before running the
same input again with `main.py` (otherwise ARCOPY offers to resume the saved run and the answers no
longer match the questions); `run_all.py` deletes it by itself. A baseline run takes a few seconds,
a Bayesian run 2 to 7 minutes and a screening run 3 to 30 minutes (2 CPU cores); the 20 runs take
about 100 minutes one after the other.

## Reference results

`reference_results/summary.csv` has one row per run. For each run, `reference_results/<run>/` holds
`retrofit_global_optimization.csv` (every strategy analysed, with cost, drift, utilization and
feasibility) and, for the Bayesian runs, `smart_search_history.csv` (every trial of the search).
The baseline runs have `modal_periods.csv` and `story_drifts.csv`. `F4-5B_bayes`, the step-by-step
example of the paper (Section 4.2), also has the other result tables and the retrofit report.
The files were produced with ARCOPY v1.3.0 (Python 3.13, OpenSeesPy 3.8, Optuna 5.1). The Bayesian
search has a fixed seed, so the same Optuna version gives the same trials.

### Existing frames and selected strategy

| Frame | Members failing (existing) | Selected run | New walls (bay:storeys, thickness) | Jacketed members | Largest utilization after retrofit | Cost (EUR) |
|---|---|---|---|---|---|---|
| F4-5B | 26 of 44 | `F4-5B_bayes` | B3:S1–3, 150 mm | 20 (20 beams) | 0.990 | 44,454 |
| F6-5B | 36 of 66 | `F6-5B_bayes` | B4:S3–6 + B5:S1–4, 150 mm | 31 (30 beams, 1 column) | 0.996 | 87,548 |
| F8-5B | 46 of 88 | `F8-5B_bayes3` | B2:S2–8 + B3:S1–8 + B4:S1–8, 150 mm | 41 (40 beams, 1 column) | 0.990 | 162,089 |
| F4-7B | 36 of 60 | `F4-7B_bayes` | B6:S1–4, 150 mm | 30 (28 beams, 2 columns) | 0.997 | 57,177 |
| F6-7B | 50 of 90 | `F6-7B_bayes` | B2:S1–6 + B3:S2–4, 150 mm | 43 (41 beams, 2 columns) | 0.995 | 110,099 |
| F8-7B | 62 of 120 | `F8-7B_bayes3` | B3:S1–7 + B4:S2–8 + B5:S3–8, 150 mm | 56 (53 beams, 3 columns) | 0.999 | 178,715 |

The selected strategy is the cheapest feasible one of the runs of each frame. All the existing
frames meet the drift limit, so the retrofit is governed by the member strength.

### Screening and Bayesian search

| Frame | Run | Method | Wall segments | Strategies analysed | Feasible | Cheapest feasible (EUR) | Found at analysis |
|---|---|---|---|---|---|---|---|
| F4-5B | `F4-5B_screen` | Screening | up to 2 | 41 | 20 | 45,588 | – |
| F4-5B | `F4-5B_bayes` | Bayesian | up to 2 | 37 | 20 | 44,454 | 9 |
| F6-5B | `F6-5B_screen` | Screening | up to 2 | 41 | 0 | none (cheapest analysed: 79,855, not feasible) | – |
| F6-5B | `F6-5B_bayes` | Bayesian | up to 2 | 40 | 6 | 87,548 | 36 |
| F8-5B | `F8-5B_screen` | Screening | up to 2 | 41 | 0 | none (cheapest analysed: 108,288, not feasible) | – |
| F8-5B | `F8-5B_bayes` | Bayesian | up to 2 | 40 | 0 | none (cheapest analysed: 108,288, not feasible) | – |
| F8-5B | `F8-5B_bayes3` | Bayesian | up to 3 | 40 | 3 | 162,089 | 8 |
| F4-7B | `F4-7B_screen` | Screening | up to 2 | 41 | 21 | 57,177 | – |
| F4-7B | `F4-7B_bayes` | Bayesian | up to 2 | 28 | 16 | 57,177 | 9 |
| F6-7B | `F6-7B_screen` | Screening | up to 2 | 41 | 0 | none (cheapest analysed: 100,485, not feasible) | – |
| F6-7B | `F6-7B_bayes` | Bayesian | up to 2 | 40 | 3 | 110,099 | 33 |
| F8-7B | `F8-7B_screen` | Screening | up to 2 | 41 | 0 | none (cheapest analysed: 156,639, not feasible) | – |
| F8-7B | `F8-7B_bayes` | Bayesian | up to 2 | 40 | 0 | none (cheapest analysed: 155,761, not feasible) | – |
| F8-7B | `F8-7B_bayes3` | Bayesian | up to 3 | 40 | 4 | 178,715 | 21 |

For the screening, the strategies analysed are the 40 layouts with the smallest drift and the
frame without new walls. On the six-storey frames the screening finds no feasible strategy and
the Bayesian search does. On the eight-storey frames two wall segments are not enough for either
method; with three segments the Bayesian search finds a feasible strategy, starting from its
physics-guided trial with three adjacent full-height walls. The screening with three segments is
not included: it would have to check about 206,000 layouts for F8-5B and 619,000 for F8-7B.

### F4-5B, the step-by-step example

The Bayesian search makes 97 proposals and analyses 37 strategies, 20 of them feasible. The
cheapest is found at analysis 9: a 150 mm wall in bay 3 over storeys 1 to 3 and 20 jacketed
beams, 44,454 EUR (wall 15,136 + jackets 29,318), with T1 = 0.25 s after the retrofit.
An evaluation of all 4,301 wall layouts of this frame (about 3 hours) confirms that this is the
cheapest of them. The jackets of each layout come from the iterative jacketing rule, so it is the
optimum over the wall layouts for this jacketing procedure.

## The 115 answers of `F4-5B_bayes.txt`

| # | Prompt | Answer |
|---|---|---|
| 1 | Run name / folder [run_&lt;date&gt;]: | `F4-5B_bayes` |
| 2 | Structure type (RC / Steel): | `rc` |
| 3 | Number of floors: | `4` |
| 4 | Floor height [m]: | `3.0` |
| 5 | Number of bays: | `5` |
| 6 | Length of bay 1 [m]: | `5` |
| 7 | Length of bay 2 [m]: | `5` |
| 8 | Length of bay 3 [m]: | `6` |
| 9 | Length of bay 4 [m]: | `5` |
| 10 | Length of bay 5 [m]: | `5` |
| 11 | Add a shear wall panel? (y/n): | `n` |
| 12 | Select option (1/2/3) [1]: | `3` |
| 13 | Width b [mm] [300.0]: | `300` |
| 14 | Height h [mm] [500.0]: | `400` |
| 15 | Concrete cover [mm] [30.0]: | `30` |
| 16 | Top bars number [3]: | `3` |
| 17 | Top bar diameter [mm] [16.0]: | `16` |
| 18 | Bottom bars number [3]: | `3` |
| 19 | Bottom bar diameter [mm] [16.0]: | `16` |
| 20 | Stirrup/tie diameter [mm] [8.0]: | `8` |
| 21 | Stirrup/tie spacing s [mm] [150.0]: | `150` |
| 22 | Number of stirrup/tie legs [2]: | `2` |
| 23 | Use smaller stirrup spacing in critical end zones? (y/n) [n]: | `n` |
| 24 | Width b [mm] [400.0]: | `400` |
| 25 | Height h [mm] [400.0]: | `400` |
| 26 | Concrete cover [mm] [35.0]: | `35` |
| 27 | Top-face bars including corners [4]: | `4` |
| 28 | Top-face bar diameter [mm] [16.0]: | `16` |
| 29 | Bottom-face bars including corners [4]: | `4` |
| 30 | Bottom-face bar diameter [mm] [16.0]: | `16` |
| 31 | Left-side intermediate bars excluding corners [2]: | `2` |
| 32 | Right-side intermediate bars excluding corners [2]: | `2` |
| 33 | Side-face bar diameter [mm] [16.0]: | `16` |
| 34 | Stirrup/tie diameter [mm] [8.0]: | `8` |
| 35 | Stirrup/tie spacing s [mm] [100.0]: | `100` |
| 36 | Number of stirrup/tie legs [4]: | `4` |
| 37 | Use smaller stirrup spacing in critical end zones? (y/n) [n]: | `n` |
| 38 | Select option (1/2): | `2` |
| 39 | Concrete fck [MPa]: | `25` |
| 40 | Steel fyk [MPa]: | `450` |
| 41 | Change gamma_c? (default 1.5) (y/n): | `n` |
| 42 | Change gamma_s? (default 1.15) (y/n): | `n` |
| 43 | Change alpha_cc? (default 0.85) (y/n): | `n` |
| 44 | Select (1/2) [2]: | `2` |
| 45 | Knowledge level [KL1/KL2/KL3] (default KL2): | `KL2` |
| 46 | Plot selected RC beam/column sections? (y/n) [y]: | `n` |
| 47 | Concrete E [MPa]: | `31000` |
| 48 | Concrete density [kg/m3]: | `2500` |
| 49 | Select (1/2) [1]: | `1` |
| 50 | G distributed load on floor 1 [kN/m] (magnitude, applied downward): | `23.0` |
| 51 | G distributed load on floor 2 [kN/m] (magnitude, applied downward): | `23.0` |
| 52 | G distributed load on floor 3 [kN/m] (magnitude, applied downward): | `23.0` |
| 53 | G distributed load on roof [kN/m] (magnitude, applied downward): | `23.0` |
| 54 | Number of live load cases (0 = none) [0]: | `1` |
| 55 | Live case 1 name [live01]: | (Enter = default) |
| 56 | live01: load [kN/m] (magnitude, applied downward): | `2.5` |
| 57 | live01: floors (all \| 1,3 \| 1-3; roof = 4) [all]: | `1` |
| 58 | live01: bays (all \| 1,3 \| 2-4) [all]: | `3` |
| 59 | Add more beams to live01 with another load? (y/n) [n]: | `n` |
| 60 | Choose profile (1/2) [1]: | (Enter = default) |
| 61 | W+ distributed load on the storey 1 column (ground storey) [kN/m] (0 if none): | `1.25` |
| 62 | W+ distributed load on the storey 2 column [kN/m] (0 if none): | `3.75` |
| 63 | W+ distributed load on the storey 3 column [kN/m] (0 if none): | `6.25` |
| 64 | W+ distributed load on the storey 4 column (top storey) [kN/m] (0 if none): | `8.75` |
| 65 | Apply at the same time on the right (leeward) exterior columns the same W+ loads, in the same direction, multiplied by a coefficient? (y/n) [n]: | `n` |
| 66 | Use the same storey loads (and leeward coefficient) for W-, mirrored? (y/n) [y]: | (Enter = default) |
| 67 | Add extra load? (y/n): | `n` |
| 68 | Keep these groups? (y/n) [y]: | `n` |
| 69 | Group name: | `Q_LIN` |
| 70 | Type: 1) Linear Add  2) Range Add  3) Envelope [1]: | `1` |
| 71 | Members (W+; W-; live01), separated by ; [all live cases]: | (Enter = default) |
| 72 | Group name (Enter = stop): | (Enter = default) |
| 73 | Variable load Q (Q_LIN / none) [Q_LIN]: | (Enter = default) |
| 74 | Live load in the seismic mass (Q_LIN / none) [Q_LIN]: | (Enter = default) |
| 75 | Seismic mass = SW + G + psi x that group (all members loaded);  psi [0.3]: | (Enter = default) |
| 76 | Add another load combination of your own? (y/n) [n]: | (Enter = default) |
| 77 | Number of sections along each BEAM [Enter = built-in]: | (Enter = default) |
| 78 | Number of sections along each COLUMN [Enter = built-in]: | (Enter = default) |
| 79 | Run EC2 checks for ULS combinations? (y/n): | `y` |
| 80 | Show colored verification table? (y/n): | `n` |
| 81 | Show utilization colored on structure? (y/n): | `n` |
| 82 | Select option (1/2/3): | `3` |
| 83 | Go to dynamic analysis now? (y/n): | `y` |
| 84 | Additional lumped mass per floor [kg] (0 if none): | `0` |
| 85 | Use rigid diaphragm per floor? (y/n): | `n` |
| 86 | Cracked stiffness factor EI_eff/EI for seismic analyses (EC8 4.3.1(7)) [default 0.5]: | (Enter = default) |
| 87 | Number of eigen modes to compute [max 72]: | `4` |
| 88 | Plot floor mode-shape curves? (y/n): | `n` |
| 89 | Plot full deformed structure for each mode? (y/n): | `n` |
| 90 | Run EC8 response spectrum analysis (design spectrum, q-factor)? (y/n): | `y` |
| 91 | Spectrum: 1) EN 1998-1   2) NTC 2008 (Italy) [1]: | `2` |
| 92 | ag/g on rock [e.g. 0.193493]: | `0.193493` |
| 93 | F0, maximum amplification factor [e.g. 2.443832]: | `2.443832` |
| 94 | Tc* [s] [e.g. 0.33327]: | `0.33327` |
| 95 | Ground type [A/B/C/D/E]: | `B` |
| 96 | Topographic category [T1/T2/T3/T4] (default T1): | `T1` |
| 97 | Damping ratio ξ as decimal [default 0.05]: | `0.05` |
| 98 | Modal combination [CQC/SRSS] (default CQC): | `CQC` |
| 99 | Behaviour factor q [default 1.50]: | (Enter = default) |
| 100 | Plot EC8 spectrum? (y/n): | `n` |
| 101 | Select (1/2/3) [1]: | `1` |
| 102 | Reduction factor ν [default 0.50 (importance I-II), 0.40 for III-IV]: | (Enter = default) |
| 103 | Run linear time-history analysis? (y/n): | `n` |
| 104 | Build global performance summary (static + RSA + TH)? (y/n): | `y` |
| 105 | Run retrofit optimization? (y/n): | `y` |
| 106 | Max wall segments per layout (1-3) [default 2]: | (Enter = default) |
| 107 | Choose search method [default 2]: | `2` |
| 108 | Number of smart trials (complete strategies) [default 60]: | `40` |
| 109 | Use default retrofit cost parameters? (y/n) [y]: | (Enter = default) |
| 110 | Max ratio jacketed / original section dimension [default 1.5]: | (Enter = default) |
| 111 | Max jacket thickness [mm] [default 200]: | (Enter = default) |
| 112 | Include the EC8 seismic action in the retrofit (lateral force method per candidate)? (y/n) [y]: | (Enter = default) |
| 113 | Use the seismic parameters of the RSA step (q = 1.50, ag = 0.193 g)? (y/n) [y]: | (Enter = default) |
| 114 | Plot before/after RC sections for jacketed members? (y/n) [y]: | (Enter = default) |
| 115 | Generate the retrofit report (Word + Excel)? (y/n) [y]: | (Enter = default) |
