"""Cross-tradition checks: place out-of-fit national CDW figures under the
production posterior.

Check 1: Invest NI (NBS-lineage) market briefing, 2022 generation 2.75 Gt
         -> percentile under the 2022 latent posterior (not in the fit).
Check 2: An & Guo (2024) dynamic MFA, 224.08 Bt cumulative 2000-2100
         -> century-average 2.22 Gt/yr vs posterior mean 2020-2024.

Inputs: results/china_theta_draws_v2c.npy (T x S, log 10^4 t, years 2000-2024)
        results/china_national_posterior_v2c.csv
Output: results/cross_tradition_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

draws = np.load(RES / "china_theta_draws_v2c.npy")   # log(10^4 t), 2000..2024
YEARS = list(range(2000, 2025))

v2022 = draws[YEARS.index(2022)]
pct_investni = float((v2022 < np.log(275000.0)).mean())   # 2.75 Gt = 275000*10^4 t
med2022 = float(np.exp(np.median(v2022))) / 1e5

post = pd.read_csv(RES / "china_national_posterior_v2c.csv")
mean_2020_2024 = float(post[(post.year >= 2020) & (post.year <= 2024)]
                       ["median_wt"].mean()) / 1e5
an_avg = 224.08 / 101

out = {
    "investni2022": {
        "value_Gt": 2.75, "year": 2022,
        "posterior_percentile": pct_investni,
        "posterior_median_Gt": med2022,
        "ratio_to_median": 2.75 / med2022,
        "broad_official_ratio_v2c": 1.2507,
        "note": "NBS-lineage market estimate; not in the fit",
    },
    "an2024century": {
        "cumulative_Bt_2000_2100": 224.08,
        "century_average_Gt_per_yr": an_avg,
        "posterior_mean_Gt_2020_2024": mean_2020_2024,
        "difference_pct": (mean_2020_2024 - an_avg) / an_avg * 100,
        "note": "dynamic MFA; not in the fit",
    },
}
(RES / "cross_tradition_check.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
