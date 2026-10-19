"""W3b: EU rolling-origin probabilistic forecast evaluation.

Two windows in one fit each (states span all 7 waves; observations truncated):
  W1: train <= 2018 -> predict 2020
  W2: train <= 2020 -> predict 2022
All 27 EU countries, both calibers (TOTAL_HH anchor + F) as observations.
Metrics per country: 90% PI coverage of the held-out TOTAL_HH value, interval
score (Winkler), and median error vs persistence baseline (last observed value).

Output: results/eu_forecast_W1.csv, eu_forecast_W2.csv (+ console summary).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eu_testbed import EU27, load_panel, fit, theta_mixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def run_window(df_all, years, train_last, pred_year, tag, seed):
    years = [int(y) for y in years]
    t_pred = years.index(pred_year)
    df_tr = df_all[df_all["time"] <= train_last].reset_index(drop=True)
    C = len(EU27)
    T = len(years)
    first_year = years[0]
    lvl = (df_tr[(df_tr["m"] == 0) & (df_tr["time"] == first_year)]
           .set_index("c")["y"])
    med = df_tr[df_tr["m"] == 0]["y"].median()
    level0 = [lvl.get(c, med) for c in range(C)]
    print(f"[{tag}] train<= {train_last}, predict {pred_year}: {len(df_tr)} obs",
          flush=True)
    smp = fit(df_tr, C, T, level0, seed=seed, warm=800, samp=800)
    draws = theta_mixture(smp, df_tr, C, T, thin=2)
    truth = (df_all[(df_all["m"] == 0) & (df_all["time"] == pred_year)]
             .set_index("c")["y"])
    rows = []
    for c, geo in enumerate(EU27):
        if c not in truth.index:
            continue
        tru = float(np.exp(truth[c]))
        d = np.exp(draws[c * T + t_pred, :])
        lo, medd, hi = np.quantile(d, [0.05, 0.5, 0.95])
        # persistence: last observed TOTAL_HH for this country
        hist = df_tr[(df_tr["c"] == c) & (df_tr["m"] == 0)].sort_values("time")["y"]
        pers = float(np.exp(hist.iloc[-1])) if len(hist) else np.nan
        alpha = 0.10
        iscore = (hi - lo) + (2 / alpha) * (lo - tru) * (tru < lo) \
            + (2 / alpha) * (tru - hi) * (tru > hi)
        rows.append(dict(geo=geo, year=pred_year, truth_Mt=tru / 1e6,
                         pred_Mt=medd / 1e6, lo=lo / 1e6, hi=hi / 1e6,
                         covered=bool(lo <= tru <= hi),
                         int_score=iscore / 1e6,
                         model_err_pct=(medd - tru) / tru * 100,
                         persistence_err_pct=(pers - tru) / tru * 100))
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"eu_forecast_{tag}.csv", index=False)
    cov = out["covered"].mean()
    print(f"[{tag}] coverage={cov:.3f} ({out['covered'].sum()}/{len(out)})  "
          f"median|err| model={out['model_err_pct'].abs().median():.1f}%  "
          f"persistence={out['persistence_err_pct'].abs().median():.1f}%  "
          f"mean IS={out['int_score'].mean():.1f} Mt")
    return out


def main():
    df, years = load_panel()
    run_window(df, years, 2018, 2020, "W1", seed=31)
    run_window(df, years, 2020, 2022, "W2", seed=32)


if __name__ == "__main__":
    main()
