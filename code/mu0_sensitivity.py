"""P1: mu0 prior sensitivity for the China system.

Reruns the production China model with the initial-level prior standard
deviation tripled (0.6 -> 1.8) through the model's parameter interface,
quantifying the influence of centering mu0 at the first digitized WGR
observation. Writes the *_v2s outputs and prints the maximum shift of
posterior medians relative to the production v2c run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import w3_china_national_v2c as v2c  # noqa: E402

v2c.main(mu0_scale=1.8, seed=46, out_suffix="v2s")

base = pd.read_csv(ROOT / "results" / "china_national_posterior_v2c.csv")
sens = pd.read_csv(ROOT / "results" / "china_national_posterior_v2s.csv")
shift = ((sens["median_wt"] - base["median_wt"]) / base["median_wt"] * 100).abs()
print(f"\nSENSITIVITY (mu0 sd 0.6 -> 1.8): max |median shift| = "
      f"{shift.max():.2f}% (year {int(base.loc[shift.idxmax(), 'year'])}), "
      f"mean = {shift.mean():.2f}%")
