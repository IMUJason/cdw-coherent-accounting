"""R5-10 / §3.2: capacity decision under the CORRECT annual-loss target.

Old (wrong) target: average G over 2026-2030 first, then compute mismatch on
the path mean --- convexity makes this underestimate the annual loss.
Correct target for a fixed annual capacity C:

    L(C) = k*C + (1/5) * sum_{t=2026..2030} E[ p*(G_t-C)^+ + h*(C-G_t)^+ ]
    F_mix(C) = (1/5) * sum_t P(G_t <= C)
    C* = F_mix^{-1}[(p-k)/(p+h)]   (interior solution when p > k; else C*=0)

Point-plan baseline: capacity set at the median of the year-marginal mixture
(this choice is itself a convention; it is documented in the output).

Inputs: results/china_forecast_draws_v2c.npy  (6 x N draws, 10^4 t, 2025..2030)
Outputs: results/decision_value_annual.csv, decision_value_annual_sensitivity.csv,
         decision_value_annual_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

fut = np.load(RES / "china_forecast_draws_v2c.npy")     # 10^4 t, 2025..2030
Gy = fut[1:6, :] / 1e5                                   # Gt/yr, 2026..2030
n_draws = Gy.shape[1]

CAPEX_CNY = 4186.76e4
CAP_T = 17.52e4
CRF = 0.08 * 1.08**20 / (1.08**20 - 1)
K = CAPEX_CNY / CAP_T * CRF                              # 24.3 CNY/(t*yr)
P, H = 95.0, 25.0


def F_mix(C):
    return float(np.mean(Gy <= C))


def L(C, k, p, h):
    per_year = (p * np.maximum(Gy - C, 0).mean(axis=1)
                + h * np.maximum(C - Gy, 0).mean(axis=1))
    return k * C + per_year.mean()


def solve_C(k, p, h):
    if p <= k:
        return 0.0, 0.0
    target = (p - k) / (p + h)
    # quantile of the year-marginal mixture: pool all (t, s) values
    pooled = Gy.reshape(-1)
    C = float(np.quantile(pooled, target))
    return C, target


def evaluate(C, k, p, h):
    short = np.maximum(Gy - C, 0)
    idle = np.maximum(C - Gy, 0)
    cost_year = k * C + p * short.mean(axis=1) + h * idle.mean(axis=1)
    return dict(C_Gt=float(C),
                cost_per_t=float(cost_year.mean() / Gy.mean()),
                shortfall_prob=float((Gy > C).mean()),
                ann_shortfall_Gt=float(short.mean(axis=0).mean()),
                ann_idle_Gt=float(idle.mean(axis=0).mean()),
                worst_year_shortfall_Gt=float(short.mean(axis=1).max()))


C_star, target = solve_C(K, P, H)
C_point = float(np.quantile(Gy.reshape(-1), 0.5))       # median of mixture
# verify closed form by grid search
grid = np.linspace(1.0, 4.5, 3500)
Lg = [L(c, K, P, H) for c in grid]
C_grid = float(grid[int(np.argmin(Lg))])
star = evaluate(C_star, K, P, H)
pt = evaluate(C_point, K, P, H)

print(f"k={K:.1f}  p={P}  h={H}   target quantile F*={target:.4f}")
print(f"C* (closed form) = {C_star:.4f} Gt ; C* (grid)    = {C_grid:.4f} Gt")
print(f"point plan (median of year-marginal mixture) = {C_point:.4f} Gt")
print(f"gap = {C_star - C_point:+.4f} Gt ({(C_star - C_point) / C_point * 100:+.1f}%)")
print(f"point : cost={pt['cost_per_t']:.2f} CNY/t  P(short)={pt['shortfall_prob']:.2f}  "
      f"ann.short={pt['ann_shortfall_Gt']:.3f}  ann.idle={pt['ann_idle_Gt']:.3f}")
print(f"Bayes : cost={star['cost_per_t']:.2f} CNY/t  P(short)={star['shortfall_prob']:.2f}  "
      f"ann.short={star['ann_shortfall_Gt']:.3f}  ann.idle={star['ann_idle_Gt']:.3f}")
print(f"saving = {pt['cost_per_t'] - star['cost_per_t']:.2f} CNY/t "
      f"({(pt['cost_per_t'] - star['cost_per_t']) / pt['cost_per_t'] * 100:.2f}%)")

# old (incorrect) path-mean target for the record
Gbar = Gy.mean(axis=0)
Fold = float(np.quantile(Gbar, target))
print(f"[record only] old path-mean target optimum = {Fold:.4f} Gt "
      f"(differs from annual target by {C_star - Fold:+.4f} Gt)")

pd.DataFrame([dict(policy="point (median of year-marginal mixture)", **pt),
              dict(policy="Bayes (annual-loss optimal)", **star)]).to_csv(
    RES / "decision_value_annual.csv", index=False)

rows = []
for p in [30.0, 37.0, 70.0, 95.0, 120.0, 140.0, 200.0]:
    for h in [2.5, 25.0, 45.0]:
        Cs, tg = solve_C(K, p, h)
        pp = evaluate(C_point, K, p, h)
        bb = evaluate(Cs, K, p, h)
        rows.append(dict(k=K, c_pen=p, c_idle=h, F_star=tg if tg else 0.0,
                         C_point=C_point, C_bayes=Cs,
                         gap_pct=(Cs - C_point) / C_point * 100,
                         cost_point=pp["cost_per_t"], cost_bayes=bb["cost_per_t"],
                         saving_CNY_per_t=pp["cost_per_t"] - bb["cost_per_t"],
                         saving_pct=(pp["cost_per_t"] - bb["cost_per_t"])
                         / pp["cost_per_t"] * 100,
                         shortfall_bayes=bb["shortfall_prob"]))
sens = pd.DataFrame(rows)
sens.to_csv(RES / "decision_value_annual_sensitivity.csv", index=False)
print("\ngap (% of point plan):")
print(sens.pivot_table(index="c_pen", columns="c_idle",
                       values="gap_pct").round(1).to_string())
print("\nsaving (%):")
print(sens.pivot_table(index="c_pen", columns="c_idle",
                       values="saving_pct").round(1).to_string())

meta = dict(k_cap=float(K), capex_cny=float(CAPEX_CNY), cap_t_per_yr=float(CAP_T),
            crf=float(CRF), n_draws=int(n_draws),
            C_point=float(C_point), C_star=float(C_star), C_grid=float(C_grid),
            F_star=float(target),
            cost_point=pt["cost_per_t"], cost_bayes=star["cost_per_t"],
            threshold_c_pen_star_low=float(2 * K + 2.5),
            threshold_c_pen_star_high=float(2 * K + 45.0),
            old_path_mean_optimum=float(Fold),
            note="annual-loss target per R5 §3.2; point plan = median of the "
                 "year-marginal predictive mixture (documented convention)")
(RES / "decision_value_annual_meta.json").write_text(
    json.dumps(meta, indent=2), encoding="utf-8")
print("\nsaved decision_value_annual*.csv/.json")
