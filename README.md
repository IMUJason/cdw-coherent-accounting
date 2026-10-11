# cdw-coherent-accounting

Minimal reproducibility package for a study that reconciles contradictory
published estimates of construction and demolition waste (CDW) generation into a
single coherent, interval-carrying series with method- and caliber-specific bias
factors. The framework is validated on the official Eurostat mineral-CDW
hierarchy (EU-27, 2010–2022) and deployed on China's fragmented national
statistics (2000–2024, forecast to 2030).

Every published figure is treated as a noisy, biased observation of a latent
true quantity; inference is fully Bayesian with the latent states analytically
collapsed (NUTS on hyperparameters, exact Gaussian conditional for the states).
Simulation-based calibration shows the collapsed scheme is calibrated where an
empirical-Bayes plug-in is badly overconfident.

## Repository layout

```
code/                            all analysis code (run order = numeric/pipeline order)
  s1_eurostat_fetch.py           fetch Eurostat env_wasgen W121 panel (both calibers) -> data/eurostat/
  build_china_obs.py             build the annotated China multi-source observation database -> data/china/
  digitize_gao_fig3.py           pixel-level digitization of Gao et al. (2025) Fig. 3 national series
  s2_sbc.py                      simulation-based calibration: oracle + empirical-Bayes plug-in arms
  s2c_sbc_uncentered.py          simulation-based calibration: collapsed fully Bayesian arm (final scheme)
  eu_testbed.py                  EU validation: dual-caliber fit, masked-state recovery, hierarchy audit
  w3_eu_forecast.py              EU rolling-origin probabilistic forecast evaluation
  w3_china_national.py           China national model, constant-drift variant (overshoot comparison)
  w3_china_national_v2c.py       China national model, final (mean-reverting growth, fixed digitization noise)
data/
  eurostat/                      raw fetched Eurostat panel (W121 mineral CDW, two calibers)
  china/                         annotated multi-source observation database + digitized series
  source_pdfs/                   empty: place the open-access Gao et al. (2025) PDF here (see below)
results/                         released numeric outputs (CSV/text) reported in the manuscript
```

## Data sources (all public)

| Source | Used by | Access |
|---|---|---|
| Eurostat `env_wasgen`, waste category W121 (mineral waste from construction and demolition), units T, calibers TOTAL_HH and F | `s1_eurostat_fetch.py` (re-fetch) / `data/eurostat/` (bundled) | https://ec.europa.eu/eurostat/databrowser/view/env_wasgen |
| MEE Solid Waste & Chemicals Management Technology Center, national disclosure-aggregation reports (2023 and 2024 data) | observation database | https://www.meescc.cn/zhxx/zxyw/202505/W020250530336582088864.pdf ; https://www.meescc.cn/zhxx/zxyw/202603/W020260317610986472151.pdf |
| MEE draft technical specification explanatory notes ("over 3.0 Gt", 2023) | observation database | https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202504/W020250415370988316064.pdf |
| Gao et al. (2025), Sustainability 17(4):1711, WGR-based series 2000–2022 | `digitize_gao_fig3.py` | DOI: 10.3390/su17041711 (open access; place PDF in `data/source_pdfs/`) |
| Zheng et al. (2017), Constr. Build. Mater. 136:405–413 (2.36 Gt/yr average 2003–2013) | observation database | DOI: 10.1016/j.conbuildmat.2017.01.055 |
| Wang et al. (2022), Waste Manag. Res. 41(4):914–923 ("approximately 2.4 billion tonnes per year") | observation database | DOI: 10.1177/0734242X221126393 |
| Industry-association figures (2014, 2017) | observation database | URLs embedded in `data/china/china_observations_v01.csv` |

The bundled `data/` files are the exact inputs used for the released results;
`data/china/china_observations_v01.csv` carries per-observation caliber,
method-family, credibility grade, source URL, and notes (57 observations,
19 sources; 30 national-caliber observations across 7 measurement processes are
used in fitting).

## Environment

Python ≥ 3.10 with:
```
numpy  scipy  pandas  jax  numpyro
```
(CPU only; no GPU required. Tested with numpyro 0.19 / jax 0.6.)

## How to reproduce

```bash
# EU panel: re-fetch (optional; bundled data already in data/eurostat/)
python code/s1_eurostat_fetch.py

# China observation database (guards included; rebuilds data/china/ CSVs)
python code/build_china_obs.py

# Digitized Gao series (requires the open-access PDF in data/source_pdfs/)
python code/digitize_gao_fig3.py

# SBC arms (Table: inference-scheme calibration)
python code/s2_sbc.py                 # ~40 min  (oracle + EB plug-in)
python code/s2c_sbc_uncentered.py     # ~2.5 h   (collapsed full Bayes, both regimes)

# EU validation
python code/eu_testbed.py             # ~40 min  (dual-caliber fit, masked recovery, hierarchy audit)
python code/w3_eu_forecast.py         # ~1 h     (rolling-origin forecasts)

# China national models
python code/w3_china_national.py      # ~8 min   (constant-drift comparison)
python code/w3_china_national_v2c.py  # ~10 min  (final; writes *_v2c outputs)
```

Runtimes on a single modern CPU core (JAX may spawn additional threads).

## Provenance map (script -> released results)

| Released file in `results/` | Produced by |
|---|---|
| `s1_coverage_report.txt` | `s1_eurostat_fetch.py` |
| `s2_sbc_summary.csv` | `s2_sbc.py` |
| `s2b_sbc_summary.csv`, `s2c_sbc_summary_*.csv` | `s2c_sbc_uncentered.py` (s2b = centered-parameterization check) |
| `e3_hierarchy_check.csv`, `e2_mask_recovery_*.csv` | `eu_testbed.py` |
| `eu_forecast_W1.csv`, `eu_forecast_W2.csv` | `w3_eu_forecast.py` |
| `china_*_v2c.csv` | `w3_china_national_v2c.py` |

The China constant-drift variant (`w3_china_national.py`) is included to
reproduce the overshoot comparison (median 4.1 Gt by 2030) that motivates the
mean-reverting dynamics.

## Notes

- Observation equation: `y = B_m * theta + b_m + eps`, with one anchor caliber
  per system (B=1, b=0); `b_m` reads as a log-scale coverage/offset factor.
  All factors are relative to the anchor.
- The digitized series' noise is fixed at 0.07 on the log scale
  (±5% digitization error plus model discrepancy); see the manuscript appendix.
- `data/source_pdfs/` is intentionally empty: publisher PDFs are not
  redistributed. Download DOI 10.3390/su17041711 to run the digitization step;
  all other steps run from the bundled data alone.

If you use this package, please cite the associated manuscript and the data
sources listed above.
