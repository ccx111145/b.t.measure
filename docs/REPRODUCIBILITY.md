# Reproducibility map

Every number in the manuscript is regenerated from the files in this repository. Run everything
from the repository root.

```bash
python -m pip install -r requirements.txt
python dev/selfcheck.py      # 51 assertions, a few minutes
python dev/final_check.py    # manuscript consistency
```

## Item-by-item map

| Manuscript item | Script | Output |
|---|---|---|
| Table 1 — evidence update rules | `robot/feasible.py` | specification table, no data |
| Table 2, Fig. 2 — baselines | `exp/compare.py` → `exp/fig_vector.py` | `tab/compare_q3.csv`, `tab/compare_q4.csv` (+ `*_stats.json`) |
| Table 3 — planning isolation | `exp/planning.py` | `tab/planning_q3.csv`, `tab/planning_q4.csv` |
| Table 4 — ablation | `exp/ablation.py` | `tab/ablation_q3.csv`, `tab/ablation_q4.csv` |
| Oracle ratio | `exp/oracle.py` | `tab/oracle_q3.csv`, `tab/oracle_q4.csv` |
| Table 5 — bearing error | `exp/rf_chain.py` | `tab/rf_bearing_error.csv` |
| Table 6 — end-to-end array chain | `exp/rf_end2end.py` | `tab/rf_end2end.csv` |
| Table 7 — CRLB | `exp/crlb_check.py` | stdout |
| Table 8, §6.11 — obstructed workspace | `exp/nonconvex_consistent.py` | `tab/nonconvex_eval.csv` |
| Table 9 — antenna patterns | `exp/antenna_pattern.py` | stdout |
| Table 10 — coherent ghost | `exp/outlier.py` | `tab/outlier_q4.csv` |
| Table 11 — duty cycle | `exp/reliability.py` (duty slice) | `tab/reliability_duty.csv` |
| Table 12 — operating points; Fig. 5 | `exp/reliability.py` → `exp/fig_reliability.py` | `tab/reliability_all.json`, `figs/reliability_map.pdf` |
| Table 13 — non-ideal channels | `exp/outlier.py` | `tab/outlier_q4.csv` |
| Table 7 / §6.5 — array chain | `exp/rf_chain.py`, `exp/crlb_check.py` | see above |
| Main runs; Figs. 1, 3, 4 | `exp/run_q3.py` (runs), `exp/generalize.py` (sweeps), `exp/fig_vector.py` (plots) | `tab/q3_runs_*.csv`, `tab/generalize_q*.csv`, `figs/en_*.pdf` |
| Obstacle repair network | `exp/nonconvex_run.py`, `exp/kinematics.py` | `tab/nonconvex_net.npy` |
| Covering-number verification | `exp/covering_exact.py`, `exp/covering_number.py` | stdout |

`exp/plot_style.py` holds the shared figure style; `exp/fig_*.py` are the plotting scripts.

## Independent verification of the theory

These scripts check the paper's theoretical claims by methods that do **not** share code with the
implementation that produces the results:

| Script | What it checks |
|---|---|
| `exp/verify_exact.py` | the exact detectability predicate against the closed-form condition |
| `exp/verify_cover.py` | covering radius and the six-point lower bound |
| `exp/verify_continuous.py` | the detectability condition on the continuum |
| `exp/verify_conservative.py` | the conservative cell certificate, and the residual set |
| `exp/verify_adaptive.py` | adaptive refinement does not close the residual |
| `exp/nonconvex_bounds.py`, `exp/nonconvex_prune.py` | the obstructed-case lower bounds |
| `dev/verify_thm2b.py` | the second-measurement feasibility set |
| `dev/selfcheck.py` | **51 assertions** covering all of the above plus the end-to-end runs |

`dev/test_*.py` are six unit-test suites (geometry, selector, simulator, policy, Q4 policy, CLI).

## The vanishing residual in the continuum certificate

The enclosure condition is proved analytically almost everywhere and verified numerically on the
residual with a 15° margin. The conservative cell certificate leaves 32.8 m² at 0.63 m effective
resolution and 9.4 m² at 0.31 m, out of 1.018 × 10⁷ m² — about 3 × 10⁻⁶. Every residual cell
intersects some circle ∂B(Pᵢ, R_min), so the uncertified set shrinks to zero measure with the mesh.

`exp/verify_conservative.py` and `exp/verify_adaptive.py` reproduce both statements.

## Manuscript sources

`paper/paper_ras.tex` (Elsevier `elsarticle`) and `paper/paper_sci.tex` (IEEE `IEEEtran`) are
content-identical. Build with:

```bash
cd paper && pdflatex paper_ras.tex   # run three times for cross-references
```

`elsarticle.cls` ships with TeX Live; no other non-standard packages are required.
