"""Contextual comparison with an excluded secondary market estimate.

The Invest NI briefing is not an independently verified NBS series. The
comparison reports the discrepancy without inferring its reporting scope.
An and Guo's century-long MFA total has no matching annual estimand here;
no numerical agreement score is computed for it.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    draws = np.load(RES / "china_theta_draws_v2c.npy", allow_pickle=False)
    if draws.ndim != 2 or draws.shape[0] != 25:
        raise ValueError(f"Unexpected production draw shape: {draws.shape}")
    year_draws = np.exp(draws[2022 - 2000]) / 1e5
    median = float(np.median(year_draws))
    out = {
        "investni2022": {
            "value_Gt": 2.75,
            "year": 2022,
            "posterior_percentile": float(np.mean(year_draws < 2.75)),
            "posterior_median_Gt": median,
            "ratio_to_median": 2.75 / median,
            "included_in_fit": False,
            "role": "excluded secondary contextual estimate",
            "source_lineage": "upstream NBS attribution not independently verified; shared inputs with fitted academic estimates cannot be excluded",
            "interpretation": "above the 90% latent interval; discrepancy does not identify reporting scope or independence",
        },
        "an2024century": {
            "cumulative_Gt_2000_2100": 224.08,
            "included_in_fit": False,
            "role": "MFA literature context only",
            "comparison_status": "not scored",
            "reason": "century cumulative building-sector estimate and 2020-2024 anchor-referenced annual states have different periods and estimands; matching annual MFA values are unavailable",
        },
    }
    (RES / "cross_tradition_check_r7.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
