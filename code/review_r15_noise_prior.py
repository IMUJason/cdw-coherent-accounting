"""Noise-prior sensitivity arm for the China reconstruction (R15).

Doubles only the log-noise prior standard deviation (0.4 -> 0.8), keeping
the base slope and offset settings; all other model settings unchanged.
Writes results under results/review_r15_noise_prior/ and compares against
the archived base arm in results/review_r7/prior/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import review_r7_prior_sensitivity as prior
import w3_china_national_v2c as cn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_r15_noise_prior"
BASE = ROOT / "results" / "review_r7" / "prior"


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    prior.OUT = OUT
    df = cn.load_obs()
    theta_ref = prior.load_fixed_theta_ref()
    arm = {
        "name": "noise_wide",
        "logB_scale": 0.25,
        "offset_scale": 0.4,
        "log_sd_prior_sd": 0.8,
        "sd_center_multiplier": 1.0,
        "seed": prior.BASE_SEED + 3,
    }
    report = prior.fit_arm(arm, df, theta_ref)

    base = pd.read_csv(BASE / "base_source_ratios.csv").set_index("source")
    wide = pd.read_csv(OUT / "noise_wide_source_ratios.csv").set_index("source")
    rows = []
    for source in base.index:
        if source == cn.ANCHOR:
            continue
        row = {"source": source}
        for label, column in (("median", "ratio_median"), ("p05", "ratio_p05"), ("p95", "ratio_p95")):
            b = float(base.loc[source, column])
            w = float(wide.loc[source, column])
            row[f"base_{label}"] = b
            row[f"noise_wide_{label}"] = w
            row[f"pct_change_{label}"] = 100.0 * (w / b - 1.0)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "noise_vs_base_source_ratios.csv", index=False)

    base_years = pd.read_csv(BASE / "base_key_years.csv").set_index("year")
    wide_years = pd.read_csv(OUT / "noise_wide_key_years.csv").set_index("year")
    years = {}
    for year in (2023, 2024, 2030):
        years[str(year)] = {
            "base": base_years.loc[year, ["median_Gt", "p05_Gt", "p95_Gt"]].tolist(),
            "noise_wide": wide_years.loc[year, ["median_Gt", "p05_Gt", "p95_Gt"]].tolist(),
        }
    summary = {
        "arm": "noise_wide",
        "change": "log_sd_m prior standard deviation doubled from 0.4 to 0.8; centers and all other settings unchanged",
        "comparison_base": "archived base arm from results/review_r7/prior",
        "max_abs_median_pct_change": float(comparison["pct_change_median"].abs().max()),
        "max_abs_endpoint_pct_change": float(comparison[["pct_change_p05", "pct_change_p95"]].abs().max().max()),
        "source_ratios": rows,
        "key_years": years,
        "run": report,
    }
    (OUT / "noise_prior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "max_abs_median_pct_change", "max_abs_endpoint_pct_change", "key_years")}, indent=2))


if __name__ == "__main__":
    main()
