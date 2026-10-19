"""P2: residual-bootstrap AR(1) baseline with intervals, EU rolling windows.

For each EU-27 country and window (train<=2018 -> 2020; train<=2020 -> 2022):
fit AR(1) with intercept on the differenced log TOTAL_HH training series by OLS,
simulate 2000 one-step (biennial-wave) paths by resampling residuals, and form
the 90% predictive interval for the held-out value. Reports coverage, median
absolute error, and mean interval score alongside the model's.
Output: results/ar_bootstrap_baseline.csv + console summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR",
        "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
        "SE", "SI", "SK"]
RNG = np.random.default_rng(7)

tot = pd.read_csv(ROOT / "data" / "eurostat" / "env_wasgen_W121_TOTAL_HH.csv")
tot = tot[tot["geo"].isin(EU27) & (tot["value"] > 0)]


def ar1_boot_pi(y_log, n_boot=2000):
    """Drift-method residual bootstrap (damped AR(1), phi = 0): the standard
    interval-capable benchmark for very short series (4-5 biennial diffs make
    unrestricted AR(1) estimation explosive)."""
    d = np.diff(y_log)
    if len(d) < 2:
        med = lo = hi = float(np.exp(y_log[-1]))
        return med, lo, hi
    dbar = d.mean()
    resid = d - dbar
    d_next = dbar + RNG.choice(resid, size=n_boot, replace=True)
    draws = np.exp(y_log[-1] + d_next)
    lo, med, hi = np.quantile(draws, [0.05, 0.5, 0.95])
    return med, lo, hi


rows = []
for tag, train_last, pred in [("W1", 2018, 2020), ("W2", 2020, 2022)]:
    mod = pd.read_csv(ROOT / "results" / f"eu_forecast_{tag}.csv")
    mod = mod.set_index("geo")
    for g in EU27:
        tr = tot[(tot["geo"] == g) & (tot["time"] <= train_last)] \
            .sort_values("time")
        truth_row = tot[(tot["geo"] == g) & (tot["time"] == pred)]
        if len(tr) < 3 or truth_row.empty:
            continue
        truth = float(truth_row["value"].iloc[0])
        med, lo, hi = ar1_boot_pi(np.log(tr["value"].to_numpy()))
        alpha = 0.10
        iscore = (hi - lo) + (2 / alpha) * (lo - truth) * (truth < lo) \
            + (2 / alpha) * (truth - hi) * (truth > hi)
        m = mod.loc[g]
        rows.append(dict(
            window=tag, geo=g, truth=truth, ar_med=med, ar_lo=lo, ar_hi=hi,
            ar_covered=bool(lo <= truth <= hi), ar_err_pct=(med - truth) / truth * 100,
            ar_IS=iscore,
            mod_covered=bool(m["covered"]), mod_err_pct=m["model_err_pct"],
            mod_IS=m["int_score"]))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "ar_bootstrap_baseline.csv", index=False)
for tag in ["W1", "W2"]:
    sub = df[df["window"] == tag]
    print(f"[{tag}] n={len(sub)}")
    print(f"  AR-bootstrap : coverage={sub['ar_covered'].mean():.3f}  "
          f"MdAPE={sub['ar_err_pct'].abs().median():.1f}%  "
          f"mean IS={sub['ar_IS'].mean() / 1e6:.1f} Mt")
    print(f"  model        : coverage={sub['mod_covered'].mean():.3f}  "
          f"MdAPE={sub['mod_err_pct'].abs().median():.1f}%  "
          f"mean IS={sub['mod_IS'].mean():.1f} Mt")
