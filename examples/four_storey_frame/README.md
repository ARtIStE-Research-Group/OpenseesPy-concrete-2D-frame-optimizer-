# Example: 4-storey, 5-bay RC frame

This is the example frame of ARCOPY v1.2.0 (the paper now uses the frames of `examples/six_frames`). `inputs.txt` holds the answers to all 74 prompts, so the whole workflow (model → static → EC2 → modal → EC8 RSA → performance summary → smart retrofit search → selected strategy → report) runs without typing.

## Frame and actions

- 4 storeys of 3.0 m; 5 bays of 5.0, 5.0, 6.0, 5.0, 5.0 m; no existing walls.
- RC sections from the built-in preset (option 2), existing building, knowledge level KL2.
- Concrete E = 30,000 MPa, density 2,500 kg/m³.
- Permanent load G: 35 kN/m on floors 1–3, 30 kN/m on the roof; wind 10 kN/m at the top (W+ and W−).
- EC8: ag = 0.25 g, ground type B, spectrum type 1, 5 % damping, γI = 1.0, CQC, q = 1.5 (default), ν = 0.5.
- Retrofit: smart (Bayesian) wall search with a budget of 40 complete analyses; default costs and jacketing limits; EC8 seismic action included with the RSA parameters.

## Run it

From the repository root, in a terminal (Anaconda Prompt / cmd, Linux or macOS):

```bash
python main.py < examples/four_storey_frame/inputs.txt
```

In Windows PowerShell:

```powershell
Get-Content examples\four_storey_frame\inputs.txt | python main.py
```

The results go to the folder `example_4storey` next to `main.py`. Delete that folder before running the example again; otherwise ARCOPY offers to resume the saved run and the answers in `inputs.txt` no longer match the questions.

When the program runs from a terminal, figure windows may stop the run until they are closed. To run without windows, set the Matplotlib backend first (`set MPLBACKEND=Agg` in cmd, `$env:MPLBACKEND="Agg"` in PowerShell, `export MPLBACKEND=Agg` on Linux/macOS); the figures are still saved in the run folder.

The run takes about 2 minutes on a laptop.

## Expected results

| Quantity | Value |
|---|---|
| Fundamental period T1 | 0.84 s |
| Smart search | 88 proposals, 36 strategies analysed, 17 feasible; stopped by convergence |
| Best strategy | full-height wall in bay 3 (storeys 1–4), 150 mm, found at analysis 8 |
| Jacketing | 22 members (20 beams and 2 columns), η max after retrofit 0.98 |
| Total cost | 53,652 EUR (walls 19,015 + jacketing 34,637) |

The search uses a fixed seed (42), so the same Optuna version gives the same trials. The files in `reference_results/` were produced by this input with ARCOPY v1.2.0 (Python 3.13, Optuna 5.0, OpenSeesPy 3.8); v1.3.0 gives the same results:

- `smart_search_history.csv`: every trial with layout, cost, feasibility, source and best cost so far;
- `retrofit_global_optimization.csv`, `retrofit_top5_alternatives.csv`: all analysed strategies and the 5 cheapest;
- `retrofit_jacketing_member_report.csv`: jacket of each member of the selected strategy;
- `modal_periods.csv`, `modal_participation.csv`, `story_drifts.csv`, `response_spectrum_drifts.csv`;
- `report/retrofit_report.docx` and `report/retrofit_report_data.xlsx`: the retrofit report;
- `report/figures/`: learning curve, search map and selected layout.

## The 74 answers

| # | Prompt | Answer |
|---|---|---|
| 1 | Run name / folder [run_&lt;date&gt;]: | `example_4storey` |
| 2 | Structure type (RC / Steel): | `rc` |
| 3 | Number of floors: | `4` |
| 4 | Floor height [m]: | `3` |
| 5 | Number of bays: | `5` |
| 6 | Length of bay 1 [m]: | `5` |
| 7 | Length of bay 2 [m]: | `5` |
| 8 | Length of bay 3 [m]: | `6` |
| 9 | Length of bay 4 [m]: | `5` |
| 10 | Length of bay 5 [m]: | `5` |
| 11 | Add a shear wall panel? (y/n): | `n` |
| 12 | Select option (1/2/3) [1]: | `2` |
| 13 | Modify default values manually? (y/n) [n]: | `n` |
| 14 | Select option (1/2): | `1` |
| 15 | Change gamma_c? (default 1.5) (y/n): | `n` |
| 16 | Change gamma_s? (default 1.15) (y/n): | `n` |
| 17 | Change alpha_cc? (default 0.85) (y/n): | `n` |
| 18 | Select (1/2) [2]: | `2` |
| 19 | Knowledge level [KL1/KL2/KL3] (default KL2): | `KL2` |
| 20 | Plot selected RC beam/column sections? (y/n) [y]: | `n` |
| 21 | Concrete E [MPa]: | `30000` |
| 22 | Concrete density [kg/m3]: | `2500` |
| 23 | Select (1/2) [1]: | (Enter = default) |
| 24 | G distributed load on floor 1 [kN/m] (magnitude, applied downward): | `35` |
| 25 | G distributed load on floor 2 [kN/m] (magnitude, applied downward): | `35` |
| 26 | G distributed load on floor 3 [kN/m] (magnitude, applied downward): | `35` |
| 27 | G distributed load on roof [kN/m] (magnitude, applied downward): | `30` |
| 28 | Number of live load cases (0 = none) [0]: | (Enter = default) |
| 29 | Choose profile (1/2) [1]: | `2` (triangular, as in v1.2.0) |
| 30 | Top wind intensity for W+ [kN/m] (0 if none): | `10` |
| 31 | Apply at the same time on the right (leeward) exterior columns the same W+ loads, in the same direction, multiplied by a coefficient? (y/n) [n]: | `n` |
| 32 | Top wind intensity for W- [kN/m] (0 if none): | `10` |
| 33 | Apply at the same time on the left (leeward) exterior columns the same W- loads, in the same direction, multiplied by a coefficient? (y/n) [n]: | `n` |
| 34 | Add extra load? (y/n): | `n` |
| 35 | Add another load combination of your own? (y/n) [n]: | (Enter = default) |
| 36 | Number of sections along each BEAM [Enter = built-in]: | (Enter = default) |
| 37 | Number of sections along each COLUMN [Enter = built-in]: | (Enter = default) |
| 38 | Run EC2 checks for ULS combinations? (y/n): | `y` |
| 39 | Show colored verification table? (y/n): | `n` |
| 40 | Show utilization colored on structure? (y/n): | `n` |
| 41 | Select option (1/2/3): | `3` |
| 42 | Go to dynamic analysis now? (y/n): | `y` |
| 43 | Mass live-load factor psi [recommended 0.3]: | (Enter = default) |
| 44 | Additional lumped mass per floor [kg] (0 if none): | `0` |
| 45 | Use rigid diaphragm per floor? (y/n): | `n` |
| 46 | Cracked stiffness factor EI_eff/EI for seismic analyses (EC8 4.3.1(7)) [default 0.5]: | (Enter = default) |
| 47 | Number of eigen modes to compute [max 72]: | `6` |
| 48 | Plot floor mode-shape curves? (y/n): | `n` |
| 49 | Plot full deformed structure for each mode? (y/n): | `n` |
| 50 | Run EC8 response spectrum analysis (design spectrum, q-factor)? (y/n): | `y` |
| 51 | Spectrum: 1) EN 1998-1   2) NTC 2008 (Italy) [1]: | (Enter = default) |
| 52 | Design ground acceleration ag/g [e.g. 0.35]: | `0.25` |
| 53 | Soil class [A/B/C/D/E]: | `B` |
| 54 | Spectrum type [1 or 2] (default 1): | `1` |
| 55 | Damping ratio ξ as decimal [default 0.05]: | `0.05` |
| 56 | Importance factor γI [default 1.0]: | `1` |
| 57 | Modal combination [CQC/SRSS] (default CQC): | `CQC` |
| 58 | Behaviour factor q [default 1.50]: | (Enter = default) |
| 59 | Plot EC8 spectrum? (y/n): | `n` |
| 60 | Select (1/2/3) [1]: | `1` |
| 61 | Reduction factor ν [default 0.50 (importance I-II), 0.40 for III-IV]: | (Enter = default) |
| 62 | Run linear time-history analysis? (y/n): | `n` |
| 63 | Build global performance summary (static + RSA + TH)? (y/n): | `y` |
| 64 | Run retrofit optimization? (y/n): | `y` |
| 65 | Max wall segments per layout (1-3) [default 2]: | (Enter = default) |
| 66 | Choose search method [default 2]: | `2` |
| 67 | Number of smart trials (complete strategies) [default 60]: | `40` |
| 68 | Use default retrofit cost parameters? (y/n) [y]: | (Enter = default) |
| 69 | Max ratio jacketed / original section dimension [default 1.5]: | (Enter = default) |
| 70 | Max jacket thickness [mm] [default 200]: | (Enter = default) |
| 71 | Include the EC8 seismic action in the retrofit (lateral force method per candidate)? (y/n) [y]: | (Enter = default) |
| 72 | Use the seismic parameters of the RSA step (q = 1.50, ag = 0.250 g)? (y/n) [y]: | (Enter = default) |
| 73 | Plot before/after RC sections for jacketed members? (y/n) [y]: | (Enter = default) |
| 74 | Generate the retrofit report (Word + Excel)? (y/n) [y]: | (Enter = default) |
