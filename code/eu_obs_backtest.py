"""R5-09: EU rolling-origin backtest with BOTH interval targets.

Windows: train<=2018 -> 2020; train<=2020 -> 2022. Refits the EU two-caliber
model (production settings: 600 warmup + 600 draws) and scores the held-out
official TOTAL_HH value under two explicitly defined targets:
  latent : 90% interval of theta_{c,t*}, the anchor-caliber latent state
  obs    : 90% predictive interval of a FUTURE REPORT y = theta + eps,
           eps ~ N(0, sigma_anchor^2) with sigma_anchor drawn per posterior
           draw (measurement noise propagated).
Reports coverage, interval score, and relative interval width (width / truth)
for both targets. The default logB_F~Normal(-0.2, 0.3) is a prior on the
non-anchor log slope, not a coverage ratio; b_F is an offset and is a log ratio
only when B_F=1. The fit helper accepts an optional logB_loc, defaulting to
-0.2 to preserve the production model.
Output: results/review_r5/eu_obs_backtest.csv
"""
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eu_testbed import EU27, load_panel, make_model  # noqa: E402

RES = ROOT / "results" / "review_r5"


def fit(df_tr, C, T, level0, seed, logB_loc=-0.2):
    mcmc = MCMC(NUTS(make_model(df_tr, C, T, level0, logB_loc=logB_loc), dense_mass=False),
                num_warmup=600, num_samples=600, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples()


def country_draws(smp, df_tr, C, T, rng):
    """Per-country latent draws (C, T, S) and anchor-sigma draws (S,)."""
    take = np.arange(0, smp["mu0"].shape[0], 2)
    S = len(take)
    mu0_all = np.asarray(smp["mu0"])          # (nchain*S, C)
    drift = np.asarray(smp["drift"])
    sd_theta = np.exp(np.asarray(smp["log_sd_theta"]))
    logB_F = np.asarray(smp["logB_F"])
    b_F = np.asarray(smp["b_F"])
    log_sd_m = np.asarray(smp["log_sd_m"])    # (n, 2)
    sigma_anchor = np.exp(log_sd_m[:, 0])

    rows_by_c = {c: g for c, g in df_tr.groupby("c")}
    draws = np.empty((C, T, S))
    for si, s in enumerate(take):
        sd_t = float(sd_theta[s])
        q = 1.0 / sd_t**2
        P = np.zeros((T, T))
        P[0, 0] = q + 1.0 / 0.75**2
        P[0, 1] = -q
        for t in range(1, T - 1):
            P[t, t] = 2 * q
            P[t, t + 1] = -q
            P[t, t - 1] = -q
        P[T - 1, T - 1] = q
        P[T - 1, T - 2] = -q
        for c in range(C):
            Pp = P.copy()
            pm = float(mu0_all[s, c]) + float(drift[s]) * np.arange(T)
            r = P @ pm
            g = rows_by_c.get(c)
            if g is not None:
                for row in g.itertuples():
                    k = int(row.t)
                    if int(row.m) == 0:
                        coef = 1.0
                        off = 0.0
                        sd = float(np.exp(log_sd_m[s, 0]))
                    else:
                        coef = float(np.exp(logB_F[s]))
                        off = float(b_F[s])  # offset; log ratio only when B_F=1
                        sd = float(np.exp(log_sd_m[s, 1]))
                    w = coef / sd**2
                    Pp[k, k] += coef * w
                    r[k] += (row.y - off) * w
            cov = np.linalg.inv(Pp)
            m = cov @ r
            draws[c, :, si] = rng.multivariate_normal(m, cov)
    return draws, sigma_anchor[take]


def main():
    df, years = load_panel()
    years = [int(y) for y in years]
    T, C = len(years), len(EU27)
    rng = np.random.default_rng(777)
    rows = []
    for tag, train_last, pred, seed in [("W1", 2018, 2020, 31),
                                         ("W2", 2020, 2022, 32)]:
        t_pred = years.index(pred)
        df_tr = df[df["time"] <= train_last].reset_index(drop=True)
        lvl = (df_tr[(df_tr["m"] == 0) & (df_tr["time"] == years[0])]
               .set_index("c")["y"])
        med = df_tr[df_tr["m"] == 0]["y"].median()
        level0 = [lvl.get(c, med) for c in range(C)]
        print(f"[{tag}] fitting ({len(df_tr)} obs) ...", flush=True)
        smp = fit(df_tr, C, T, level0, seed)
        draws, sig = country_draws(smp, df_tr, C, T, rng)
        truth = (df[(df["m"] == 0) & (df["time"] == pred)]
                 .set_index("c")["y"])
        for c, geo in enumerate(EU27):
            if c not in truth.index:
                continue
            tru = float(np.exp(truth[c]))
            lat = np.exp(draws[c, t_pred, :])
            obs = np.exp(draws[c, t_pred, :] +
                         rng.normal(0.0, sig, size=draws.shape[2]))
            lo_l, hi_l = np.quantile(lat, [0.05, 0.95])
            lo_o, hi_o = np.quantile(obs, [0.05, 0.95])
            a = 0.10
            is_l = ((hi_l - lo_l) + (2 / a) * (lo_l - tru) * (tru < lo_l)
                    + (2 / a) * (tru - hi_l) * (tru > hi_l))
            is_o = ((hi_o - lo_o) + (2 / a) * (lo_o - tru) * (tru < lo_o)
                    + (2 / a) * (tru - hi_o) * (tru > hi_o))
            rows.append(dict(window=tag, geo=geo, truth_Mt=tru / 1e6,
                             lat_lo=lo_l / 1e6, lat_hi=hi_l / 1e6,
                             lat_cov=bool(lo_l <= tru <= hi_l),
                             lat_IS=is_l / 1e6,
                             obs_lo=lo_o / 1e6, obs_hi=hi_o / 1e6,
                             obs_cov=bool(lo_o <= tru <= hi_o),
                             obs_IS=is_o / 1e6))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "eu_obs_backtest.csv", index=False)
    for tag in ["W1", "W2"]:
        sub = out[out["window"] == tag]
        print(f"[{tag}] n={len(sub)}  "
              f"latent cov={sub['lat_cov'].mean():.3f} "
              f"IS={sub['lat_IS'].mean():.1f} "
              f"width={np.median(sub.lat_hi - sub.lat_lo):.1f}  |  "
              f"obs cov={sub['obs_cov'].mean():.3f} "
              f"IS={sub['obs_IS'].mean():.1f} "
              f"width={np.median(sub.obs_hi - sub.obs_lo):.1f} Mt")


if __name__ == "__main__":
    main()
