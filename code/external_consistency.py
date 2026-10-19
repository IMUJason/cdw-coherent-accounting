"""External consistency tests for the China reconstruction.

Test A (hold-out): refit the China system with the 2024 statutory anchor
observation (CN-006, 2.31 Gt) withheld, and check whether its 90% posterior
interval covers the held-out value. This is a genuine out-of-sample test of
the China model, analogous to the masked-state test on the Eurostat panel.

Test B (independent published forecast): locate Gao et al.'s (2025) published
2023 forecast (2.39 billion tonnes, obtained from their own GM(1,1)
extrapolation and NOT used in our fit) within our 2023 posterior. Reported as
a posterior percentile.

Outputs: results/external_consistency.csv + console summary.
"""
import sys
from pathlib import Path

import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_china_national_v2c as v2c  # noqa: E402

RES = ROOT / "results"
YEARS, T = v2c.YEARS, v2c.T

# ---------------- Test A: withhold the 2024 anchor ----------------
df = v2c.load_obs()
mask = df["source_id"] == "meescc_2025" if "source_id" in df else None
if mask is None:
    # load_obs re-maps source ids; identify the 2024 anchor by year + anchor source
    mask = (df["src"] == "meescc") & (df["year"] == 2024)
held = df[mask]
df_tr = df[~mask].reset_index(drop=True)
print(f"[A] withheld {len(held)} observation(s); fitting on {len(df_tr)}")

mcmc = MCMC(NUTS(v2c.make_model(df_tr, T)), num_warmup=1500, num_samples=1500,
            num_chains=1, progress_bar=False)
mcmc.run(jax.random.PRNGKey(51))
smp = mcmc.get_samples()

take = np.arange(0, smp["mu0"].shape[0], 2)
S = len(take)
draws = np.empty((T, S))
logB_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["logB_f"])[take]], 1)
b_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["b_f"])[take]], 1)
for si, s in enumerate(take):
    phis = dict(mu0=float(smp["mu0"][s]), c=float(smp["c"][s]),
                rho=float(smp["rho"][s]), sd_eta=float(smp["sd_eta"][s]),
                logB=logB_all[si], b=b_all[si],
                log_sd_m=np.concatenate([np.asarray(smp["log_sd_m_free"])[s][:1],
                                         [np.log(0.07)],
                                         np.asarray(smp["log_sd_m_free"])[s][1:]]))
    mth, cth = v2c.theta_posterior(phis, df_tr, T)
    draws[:, si] = np.random.multivariate_normal(mth, cth)

i24 = YEARS.index(2024)
d24 = np.exp(draws[i24, :])                      # 10^4 t
lo, med, hi = np.quantile(d24, [0.05, 0.5, 0.95])
truth = 231000.0                                 # 万吨 (2.31 Gt)
covered = bool(lo <= truth <= hi)
print(f"[A] 2024 posterior (2024 anchor withheld): median={med/1e5:.2f} Gt "
      f"90% PI=[{lo/1e5:.2f}, {hi/1e5:.2f}] Gt; held-out 2.31 Gt covered={covered}")

# ---------------- Test B: Gao's independent 2023 forecast ----------------
full = np.load(RES / "china_theta_draws_v2c.npy")   # T x S, log(10^4 t)
i23 = YEARS.index(2023)
pct = float((np.exp(full[i23, :]) < 239000.0).mean())  # 2.39 Gt = 239000 万吨
our_med = float(np.median(np.exp(full[i23, :])) / 1e5)
print(f"[B] our 2023 median={our_med:.2f} Gt; Gao's published 2023 forecast "
      f"2.39 Gt lies at posterior percentile {pct:.2f}")

pd.DataFrame([
    dict(test="A_holdout_2024", withheld_obs=int(len(held)), truth_Gt=2.31,
         median_Gt=med / 1e5, lo_Gt=lo / 1e5, hi_Gt=hi / 1e5, covered=covered),
    dict(test="B_Gao2023_forecast", withheld_obs=0, truth_Gt=2.39,
         median_Gt=our_med, lo_Gt=np.nan, hi_Gt=np.nan,
         covered=bool(0.05 <= pct <= 0.95)),
]).to_csv(RES / "external_consistency.csv", index=False)
print("saved results/external_consistency.csv")
