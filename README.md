# cdw-coherent-accounting

Minimal reproducibility package for a study that reconciles contradictory
published estimates of construction and demolition waste (CDW) generation into
one anchor-referenced, interval-carrying series with process-specific bias
factors. The framework is validated on the official Eurostat mineral-CDW
panel (EU-27, 2010–2022) and deployed on China's fragmented national statistics
(2000–2024, forecast to 2030).

Every published figure is treated as a noisy, biased observation of a latent
quantity; inference is fully Bayesian with the latent states analytically
collapsed (NUTS on hyperparameters, exact Gaussian conditional for the states).
Simulation-based calibration on a design matched to the production model shows
that the collapsed scheme is near nominal (0.906 at the nominal 90% level)
where a joint MAP plug-in covers 0.469, and that a naive EM plug-in collapses
structurally in this design.

## Repository layout

```
code/                            analysis code (run order = pipeline order)
  s1_eurostat_fetch.py           fetch Eurostat env_wasgen W121 panel (both calibers) -> data/eurostat/
  build_china_obs.py             build the annotated China multi-source observation database -> data/china/
  digitize_gao_fig3.py           pixel-level digitization of Gao et al. (2025) Fig. 3 national series
  eu_testbed.py                  EU validation: dual-caliber fit, masked-state recovery, hierarchy audit
  eu_obs_backtest.py             EU rolling-origin backtest with latent and future-observation targets
  w3_eu_forecast.py              EU rolling-origin forecasts (W1/W2) and persistence baseline
  w3_china_national.py           China constant-drift variant (overshoot comparison)
  w3_china_national_v2c.py       China production model (AR(1) increments, fixed digitization noise)
  sbc_production.py              matched SBC on the production China design (oracle/EM/MAP/full; misspec arm)
  sbc_rerun_emreg.py             prior-ridge EM arm, same seeds (minutes)
  sbc_rerun_emb1.py              B=1 plug-in arm, same seeds (constructively misspecified; minutes)
  review_r5_validation.py        two-chain convergence diagnostic and sensitivity cases
  decision_value_annual.py       annual-loss capacity decision and sensitivity grid
  external_consistency.py        2024-anchor holdout and same-source forecast consistency check
  mu0_sensitivity.py             initial-level prior sensitivity (mu0 sd 0.6 -> 1.8)
  make_figures_v2.py             manuscript figures 1-6
  make_fig7.py                   manuscript figure 7 (consistency + decision value)
  audit_after_r5.py              post-revision text/number audit
data/
  eurostat/                      raw fetched Eurostat panel (W121, two calibers)
  china/                         annotated observation database + digitized series
  source_pdfs/                   empty: place the open-access Gao et al. (2025) PDF here
results/                         released numeric outputs reported in the manuscript
  review_r5/                     matched SBC and dual-target backtest outputs
```

The package contains code, inputs, and numeric outputs only; the manuscript,
bibliography, and figure PDFs are not redistributed.

## Data sources (all public)

| Source | Used by | Access |
|---|---|---|
| Eurostat `env_wasgen`, waste category W121 (mineral waste from construction and demolition), units T, calibers TOTAL_HH and F | `s1_eurostat_fetch.py` (re-fetch) / `data/eurostat/` (bundled) | https://ec.europa.eu/eurostat/databrowser/view/env_wasgen |
| MEE Solid Waste & Chemicals Management Technology Center, national disclosure-aggregation reports (2023 and 2024 data) | observation database | https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf ; https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf |
| MEE draft technical specification explanatory notes ("over 3.0 Gt", 2023) | observation database | https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202504/W020250415370988316064.pdf |
| Hohhot 2024 and 2025 statutory disclosures (illustrative unit/coverage example) | observation database | URLs recorded in `data/china/china_observations_v01.csv` |
| Gao et al. (2025), Sustainability 17(4):1711, WGR-based series 2000–2022 | `digitize_gao_fig3.py` | DOI: 10.3390/su17041711 (open access; place PDF in `data/source_pdfs/`) |
| Zheng et al. (2017), Constr. Build. Mater. 136:405–413 (2.36 Gt/yr average 2003–2013) | observation database | DOI: 10.1016/j.conbuildmat.2017.01.055 |
| Wang et al. (2022), Waste Manag. Res. 41(4):914–923 ("approximately 2.4 billion tonnes per year", undated review estimate) | observation database | DOI: 10.1177/0734242X221126393 |
| Industry-association figures (2014, 2017; contextual, source pages no longer locatable) | observation database | URLs recorded in the CSV; flagged in the manuscript limitations |

The bundled `data/` files are the exact inputs used for the released results;
`data/china/china_observations_v01.csv` carries per-observation caliber,
method-family, credibility grade, source URL, and notes (58 records, 19
sources; 54 records carry numeric mass values; 30 national-caliber
observations across 7 measurement processes are used in fitting).

## Environment

Python ≥ 3.10 with:

```
numpy  scipy  pandas
jax==0.6.2  numpyro==0.19.0
matplotlib
```

CPU only; no GPU required. All scripts fix their random seeds; the production
China run uses `PRNGKey(45)` and a state-sampling seed of 45001.

## How to reproduce

```bash
# EU panel: re-fetch (optional; bundled data already in data/eurostat/)
python code/s1_eurostat_fetch.py

# China observation database (guards included; rebuilds data/china/ CSVs)
python code/build_china_obs.py

# Digitized Gao series (requires the open-access PDF in data/source_pdfs/)
python code/digitize_gao_fig3.py

# China production model (~10 min; writes *_v2c outputs)
python code/w3_china_national_v2c.py

# Two-chain convergence diagnostic (~2 min; matches the appendix)
python code/review_r5_validation.py base

# Matched SBC (~2 h for 32 replicates; ~50 min for 16 misspecification replicates)
python code/sbc_production.py main 32
python code/sbc_production.py misspec 16
python code/sbc_rerun_emreg.py
python code/sbc_rerun_emb1.py

# Annual-loss decision and sensitivity grid (seconds)
python code/decision_value_annual.py

# External consistency and prior sensitivity (~15 min each)
python code/external_consistency.py
python code/mu0_sensitivity.py

# EU validation
python code/eu_testbed.py             # ~40 min
python code/w3_eu_forecast.py         # ~1 h
python code/eu_obs_backtest.py        # ~25 min

# Figures (writes to paper/figs if present, otherwise results/figures)
python code/make_figures_v2.py
python code/make_fig7.py
```

Runtimes are for a single modern CPU core (JAX may spawn additional threads).

## Provenance map (script -> released result)

| Released file | Produced by |
|---|---|
| `results/s1_coverage_report.txt` | `s1_eurostat_fetch.py` |
| `results/e3_hierarchy_check.csv`, `e2_mask_recovery_*.csv` | `eu_testbed.py` |
| `results/eu_forecast_W1.csv`, `eu_forecast_W2.csv` | `w3_eu_forecast.py` |
| `results/review_r5/eu_obs_backtest.csv` | `eu_obs_backtest.py` |
| `results/china_national_posterior_v2c.csv` | `w3_china_national_v2c.py` |
| `results/china_forecast_v2c.csv`, `china_forecast_draws_v2c.npy` | `w3_china_national_v2c.py` |
| `results/china_source_bias_v2c.csv`, `china_theta_draws_v2c.npy` | `w3_china_national_v2c.py` |
| `results/china_*_v2s.csv` | `mu0_sensitivity.py` |
| `results/review_r5/sbc_prod_main*.csv/.json` | `sbc_production.py main 32` + `sbc_rerun_*.py` |
| `results/review_r5/sbc_prod_misspec*.csv/.json` | `sbc_production.py misspec 16` |
| `results/review_r5/china_base_run.json`, `china_base_diagnostics.csv` | `review_r5_validation.py base` (two-chain convergence diagnostic) |
| `results/decision_value_annual*.csv/.json` | `decision_value_annual.py` |
| `results/external_consistency.csv` | `external_consistency.py` |
| `results/ar_bootstrap_baseline.csv` | `ar_bootstrap_baseline.py` |

## Notes

- Observation equation: `y = B_m * theta + b_m + eps`, with one anchor caliber
  per system (`B=1`, `b=0`). All level ratios are relative to the anchor.
- The marginal likelihood includes the offset vector `b_phi`; conditional
  state draws use the exact Gaussian posterior in the manuscript appendix.
- The digitized series' noise is fixed at 0.07 on the log scale: a 0.05
  reading-precision component (about ±5%) plus a 0.05 model-discrepancy
  component.
- `decision_value_annual.py` implements the annual-loss target used in the
  manuscript: fixed annual capacity, per-year shortfall/idle costs, optimum at
  the equal-weight mixture quantile `(p-k)/(p+h)`. It is not equivalent to
  averaging the 2026–2030 path first.
- The production China run is single-chain with a fixed seed. The two-chain
  convergence diagnostic reported in the appendix is
  `python code/review_r5_validation.py base`; it writes
  `results/review_r5/china_base_run.json` and `china_base_diagnostics.csv`.
- `data/source_pdfs/` is intentionally empty: publisher PDFs are not
  redistributed. Download DOI 10.3390/su17041711 to run the digitization step;
  all other steps run from the bundled data alone.

If you use this package, please cite the associated manuscript and the data
sources listed above.
