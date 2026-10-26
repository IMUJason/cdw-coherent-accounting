# cdw-coherent-accounting

Minimal reproducibility package for a study that reconciles contradictory
published estimates of construction and demolition waste (CDW) generation
into one anchor-referenced, interval-carrying series with source-specific
bias parameters. The inference machinery is audited on the official
Eurostat mineral-CDW panel (EU-27, 2010–2022) and deployed on China's
fragmented national statistics (2000–2024, forecast to 2030).

Every published figure is treated as a noisy, potentially biased observation
of an anchor-referenced latent quantity; inference is fully Bayesian with
the latent states analytically collapsed (NUTS on hyperparameters, exact
Gaussian conditional for the states). Simulation-based calibration on a
design matched to the production model (32 replicates, eight preselected
states each) gives empirical 90% state-interval coverage of 0.902 (SE 0.017)
for a true-hyperparameter oracle, 0.457 (SE 0.058) for a hyperparameter MAP
plug-in, and 0.914 (SE 0.029) for collapsed full Bayes; a 16-replicate arm
with AR(1) serial errors in the digitized series gives 0.883 (SE 0.052).
Two EM implementations with linear-scale ridges converge in only 2/32 and
0/32 replicates; their coverage columns are retained as numerical-failure
diagnostics, not as converged comparators, and no universal EM failure is
claimed.

The package contains core data, code, and released numeric outputs only.
The manuscript, bibliography, and figures are not redistributed.

## Repository layout

```
code/                            analysis code (run order = pipeline order)
  s1_eurostat_fetch.py           fetch Eurostat env_wasgen W121 panel (both calibers) -> data/eurostat/
  build_china_obs.py             build the annotated China multi-source observation database -> data/china/
  digitize_gao_fig3.py           pixel-level digitization of Gao et al. (2025) Fig. 3 (needs source PDF)
  eu_testbed.py                  EU validation: dual-caliber fit, masked-state recovery, hierarchy audit
  eu_obs_backtest.py             EU rolling-origin backtest, latent and future-observation targets
  w3_eu_forecast.py              EU rolling-origin point forecasts and persistence baseline
  w3_china_national.py           China constant-drift variant (4.1 Gt 2030 comparison in the text)
  w3_china_national_v2c.py       China production model (AR(1) increments, fixed digitization noise)
  sbc_production.py              matched SBC on the production China design (main 32 / misspec 16)
  review_r5_validation.py        two-chain convergence diagnostic for the production China run
  review_r7_prior_sensitivity.py China measurement-prior sensitivity (half/base/double SD arms)
  review_r7_censored.py          censored-likelihood refit for the two lower-bound observations
  review_r7_loso.py              leave-one-source-out influence diagnostic
  review_r7_dynprior.py          dynamic-prior sensitivity (kappa sd, sigma_eta scale)
  review_r15_noise_prior.py      noise-prior sensitivity (log-sd prior sd 0.4 -> 0.8)
  eu_prior_sensitivity.py        EU log-slope prior-location sensitivity (two arms, full panel)
  decision_value_annual.py       annual-loss capacity decision and sensitivity grid
  external_consistency.py        deterministic 2024-anchor holdout + same-source consistency check
  cross_tradition_check.py       excluded secondary market briefing under the posterior
  mu0_sensitivity.py             initial-level prior sensitivity (mu0 sd 0.6 -> 1.8)
  ar_bootstrap_baseline.py       drift residual-bootstrap interval baseline for the EU backtest
  verify_release.py              offline release verification (data invariants, recomputed summaries)
data/
  eurostat/                      raw fetched Eurostat panel (W121, two calibers)
  china/                         annotated observation database + digitized series (derived values)
  source_pdfs/                   empty: place the open-access Gao et al. (2025) PDF here for digitization
results/                         released numeric outputs reported in the manuscript
  review_r5/                     production two-chain diagnostic + EU dual-target backtest
  review_r7/                     R7 SBC, China/EU prior sensitivities, EU numeric audit
```

## Data sources (all public)

| Source | Used by | Access |
|---|---|---|
| Eurostat `env_wasgen`, W121, calibers TOTAL_HH and F | `s1_eurostat_fetch.py` / bundled | https://ec.europa.eu/eurostat/databrowser/view/env_wasgen |
| MEE national disclosure-aggregation reports (2023, 2024 data years) | observation database | https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf ; https://www.meescc.cn/zhxx/zxyw/202603/t20260317_1146746.shtml |
| MEE draft technical specification explanatory notes ("over 3.0 Gt", 2023) | observation database | https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202504/W020250415370988316064.pdf |
| Gao et al. (2025), Sustainability 17(4):1711, WGR series 2000–2022 | `digitize_gao_fig3.py` | DOI: 10.3390/su17041711 (open access; place PDF in `data/source_pdfs/`) |
| Zheng et al. (2017), Constr. Build. Mater. 136:405–413 | observation database | DOI: 10.1016/j.conbuildmat.2017.01.055 |
| Wang et al. (2023), Waste Manag. Res. 41(4):914–923 (undated review estimate) | observation database | DOI: 10.1177/0734242X221126393 |
| Industry-association figures (2014, 2017) | observation database | URLs recorded in the CSV |
| Invest Northern Ireland market briefing (2022; excluded from fitting) | `cross_tradition_check.py` | https://www.investni.com/sites/default/files/2022-11/china-market-briefing-construction-demolition-waste-recycling.pdf |

`data/china/china_observations_v01.csv` carries per-observation caliber,
method family, source URL, and notes: 58 records from 19 sources, 54 with
numeric mass values; the fitting set is the 30 national-caliber
observations across 7 measurement processes. The derived digitized series
is bundled; the publisher PDF is not redistributed.

## Environment

Python 3.10 with:

```
numpy==2.2.6  scipy  pandas
jax==0.6.2  numpyro==0.19.0
matplotlib
pymupdf       # only for digitize_gao_fig3.py
```

CPU only. All released runs fix their seeds and enable JAX float64.
Exact floating-point reproduction across JAX/NumPyro versions or platforms
is not guaranteed; released CSV/JSON outputs are the reference.

## How to reproduce

```bash
# China observation database (guards included)
python code/build_china_obs.py

# Digitized Gao series (optional; requires the open-access PDF in data/source_pdfs/)
python code/digitize_gao_fig3.py

# China production model (~10 min, single chain, PRNGKey(45), state seed 45001)
python code/w3_china_national_v2c.py

# Two-chain convergence diagnostic for the production run
python code/review_r5_validation.py base

# Matched SBC (~2 h for 32 main replicates; ~50 min for 16 misspecification replicates)
python code/sbc_production.py main 32
python code/sbc_production.py misspec 16

# China measurement-prior sensitivity (three two-chain arms, ~1 min each)
python code/review_r7_prior_sensitivity.py

# Censored-likelihood refit, leave-one-source-out, dynamic-prior sensitivity (~2 min total)
python code/review_r7_censored.py
python code/review_r7_loso.py
python code/review_r7_dynprior.py
python code/review_r15_noise_prior.py

# Deterministic 2024-anchor holdout and same-source consistency check
python code/external_consistency.py

# Excluded secondary briefing under the posterior (seconds)
python code/cross_tradition_check.py

# Annual-loss decision and sensitivity grid (seconds)
python code/decision_value_annual.py

# EU validation (~40 min) and rolling-origin runs
python code/eu_testbed.py
python code/w3_eu_forecast.py         # archived 800+800 point-forecast run
python code/eu_obs_backtest.py        # 600+600 dual-target interval run
python code/ar_bootstrap_baseline.py

# EU log-slope prior-location sensitivity (two full-panel arms, ~1 h)
python code/eu_prior_sensitivity.py

# Offline release verification (seconds, no MCMC)
python code/verify_release.py
```

## Provenance map (script -> released result)

| Released file | Produced by |
|---|---|
| `results/s1_coverage_report.txt` | `s1_eurostat_fetch.py` |
| `results/e3_hierarchy_check.csv`, `e2_mask_recovery_*.csv` | `eu_testbed.py` |
| `results/eu_forecast_W1.csv`, `eu_forecast_W2.csv` | `w3_eu_forecast.py` (800+800 point forecasts, MdAPE source) |
| `results/review_r5/eu_obs_backtest.csv` | `eu_obs_backtest.py` (600+600 dual-target intervals; coverage/IS/width source) |
| `results/ar_bootstrap_baseline.csv` | `ar_bootstrap_baseline.py` |
| `results/china_national_posterior_v2c.csv` | `w3_china_national_v2c.py` |
| `results/china_forecast_v2c.csv`, `china_forecast_draws_v2c.npy` | `w3_china_national_v2c.py` |
| `results/china_source_bias_v2c.csv`, `china_theta_draws_v2c.npy` | `w3_china_national_v2c.py` |
| `results/china_*_v2s.csv` | `mu0_sensitivity.py` |
| `results/review_r5/china_base_*.csv/.json` | `review_r5_validation.py base` |
| `results/review_r7/sbc_main*.csv/.json/.npz/.log` | `sbc_production.py main 32` |
| `results/review_r7/sbc_misspec*.csv/.json/.npz/.log` | `sbc_production.py misspec 16` |
| `results/review_r7/prior/*` (CSV/JSON summaries) | `review_r7_prior_sensitivity.py` |
| `results/review_r7/censored/*` | `review_r7_censored.py` |
| `results/review_r7/loso/*` | `review_r7_loso.py` |
| `results/review_r7/dynprior/*` | `review_r7_dynprior.py` |
| `results/review_r15_noise_prior/*` | `review_r15_noise_prior.py` |
| `results/review_r7/eu_prior/*` (CSV/JSON summaries) | `eu_prior_sensitivity.py` |
| `results/review_r7/eu_numeric_audit_r7.json/.csv` | audit of the released EU files (provenance-distinct recomputation) |
| `results/external_consistency_r7.csv/.json/.npz` | `external_consistency.py` |
| `results/cross_tradition_check_r7.json` | `cross_tradition_check.py` |
| `results/decision_value_annual*.csv/.json` | `decision_value_annual.py` |
| `results/review_r7/release_verification.json` | `verify_release.py` |

## Notes

- Observation equation: `y = B_m * theta + b_m + eps` on the log scale, with
  one anchor process per system (`B=1`, `b=0`) defining the reference scope.
  `B_m` is a log-scale slope, not a coverage ratio; the original-scale ratio
  at a reference level is `exp((B_m-1)*theta_ref + b_m)` and changes with the
  reference level whenever `B_m != 1`.
- Marginal likelihood includes the offset vector; conditional state draws use
  the exact Gaussian posterior.
- The digitized series' noise is fixed at 0.07 on the log scale (assumed
  reading precision plus a model-discrepancy component).
- `decision_value_annual.py` implements the annual-loss target: fixed annual
  capacity, per-year shortfall/idle costs, optimum at the equal-weight
  mixture quantile `(p-k)/(p+h)`. The grid spans p = 30–200 and
  h = 2.5–45 CNY/t; all three costs are scenario assumptions.
- The prior-sensitivity base arm is a two-chain diagnostic refit with
  different seeds from the archived single-chain production run; its source
  ratios can differ from `china_source_bias_v2c.csv` by up to ~6% at an
  interval endpoint, so the sensitivity comparison is internal to its own
  base arm.
- EU numbers come from two archived runs: the 800+800 point-forecast run
  (MdAPE) and the 600+600 dual-target interval run (coverage, interval
  score, width). They are not the same MCMC draws.
- The SBC generator draws the anchor-noise scale from the production prior
  (the anchor sd is a free parameter in the model). EM arms use
  linear-scale ridges (weak: 1e-3 numerical; strong: 16 / 6.25), which are
  not the production log-slope priors.
- The deterministic holdout uses inference seed 51, state seed 51001, and
  report-noise seed 51002, and reports latent and future-report intervals
  separately.

If you use this package, please cite the associated manuscript and the data
sources listed above.
