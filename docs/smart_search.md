# Smart shear-wall search (Bayesian optimization)

At the retrofit step (step 10) ARCOPY asks:

```text
--- Wall search method ---
  1) Screening: rank all layouts with a quick drift check, then full check of the best N
  2) Smart: Bayesian optimization (TPE) - each new layout learns from all previous trials
Choose search method [default 2]:
Number of smart trials (complete strategies) [default 60]:
```

Method 1 ranks all layouts by a quick drift check and fully evaluates the best N. Method 2 is described here. It needs the `optuna` package.

## What one trial is

Every trial is a complete retrofit strategy, analysed with the same routine as the screening search:

wall layout → EC8 lateral force method → EC2/EC8 member checks → iterative concrete jacketing → total cost.

The design variables are the bays and storeys of up to two wall segments and the wall thickness (default candidates 150, 200, 250, 300 mm).

## How the search learns

1. **Physics-guided first trials**: no new wall, a full-height wall in the middle bay, walls over the failing storeys, walls in the two end bays.
2. **Exploration**: 10 random trials.
3. **TPE**: the Tree-structured Parzen Estimator (Optuna, multivariate, seed 42) splits the trials done so far into good (cheap and feasible) and bad ones and proposes the next layout where good layouts are more likely than bad ones.
4. **Constraints**: drift, member utilization (η ≤ 1) and complete feasibility (θ and jacketing limits) are passed to Optuna as constraints, so infeasible trials also teach the model where not to go.
5. **Cache**: a layout already analysed is taken from the cache and does not use the budget. The budget is the number of complete analyses.
6. **Local refinement**: when TPE proposes a layout already analysed, the next trial is a neighbour of the best strategy so far (thickness one step thinner or thicker, one storey more or less, shifted by one bay, a segment removed, or a one-storey segment added).
7. **Stop**: when the budget is used, or when max(20, budget/2) proposals in a row had nothing new to analyse.

## Example result (4-storey, 5-bay frame of `examples/four_storey_frame`)

| Search | Complete analyses | Feasible | Cheapest strategy |
|---|---|---|---|
| Screening, best 40 layouts by drift | 41 | 0 | 69,229 EUR (not feasible) |
| Smart, budget 40, seed 42 | 36 | 17 | 53,652 EUR, found at analysis 8 |

In this frame drift was not the governing problem (member utilization was), so the 40 layouts with the smallest drift contained no feasible strategy. The smart search learns from the full result (cost and all checks).

A single run depends on the random seed. Over 5 seeds with budget 40, the cheapest strategy found was 52,030 EUR (wall in bay 1 or 5, storeys 1–4, 150 mm); every run came within 5 % of it.

## Outputs

```text
<run>/smart_search_history.csv   every trial: layout, cost, feasibility, source, best so far
learning curve and search map    Spyder plots and report section on the search
report Excel sheet               Smart_search_history
```
