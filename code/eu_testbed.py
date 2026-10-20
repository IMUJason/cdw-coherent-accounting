"""W2b: EU testbed — fit the collapsed-Bayes multi-source model on REAL Eurostat data.

Two observation processes per country:
  source 1 = env_wasgen W121 TOTAL_HH  (anchor: B=1, b=0 -- "official total caliber")
  source 2 = env_wasgen W121 F         (construction-sector caliber: slope B_F, offset b_F)

Experiments:
  E1 two-caliber fit: estimate the F-caliber slope and offset. The empirical
     F/TOTAL ratio is a level ratio, r(theta)=exp((B_F-1)*theta+b_F), not B_F;
     compare it with posterior-implied ratios evaluated at the relevant theta.
  E2 mask-recovery: mask BOTH sources for the last two waves (2020, 2022) of a
     mid-size country (IE) and the largest (DE); posterior predictive vs truth;
     benchmark against persistence (last obs carried forward).
  E3 hierarchy check: posterior sum over EU27 members vs the official EU27_2020
     aggregate series (coherent truth, verified in S1).

The default prior logB_F ~ Normal(-0.2, 0.3) is on the log slope, not on the
empirical coverage ratio. b_F is a log-scale offset and is a log ratio only
when B_F=1. The default is retained; make_model/fit accept an optional prior
location for bounded sensitivity checks.

Collapsed inference identical in structure to s2_sbc_bayes.py (NUTS on hyperparams,
exact Gaussian theta | phi), with per-country level mu0_c and shared drift.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eurostat"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR",
        "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
        "SE", "SI", "SK"]
SD0 = 0.5


def load_panel():
    tot = pd.read_csv(DATA / "env_wasgen_W121_TOTAL_HH.csv")
    f = pd.read_csv(DATA / "env_wasgen_W121_F.csv")
    years = sorted(tot["time"].unique())
    tot = tot[tot["geo"].isin(EU27)].copy()
    f = f[f["geo"].isin(EU27)].copy()
    # drop zeros / nonpositive (log scale); report
    for name, d in [("TOTAL_HH", tot), ("F", f)]:
        nz = (d["value"] <= 0).sum()
        if nz:
            print(f"  note: dropped {nz} nonpositive obs in {name}")
    tot = tot[tot["value"] > 0]
    f = f[f["value"] > 0]
    tot["y"] = np.log(tot["value"])
    f["y"] = np.log(f["value"])
    tot["m"] = 0
    f["m"] = 1
    df = pd.concat([tot[["m", "geo", "time", "y"]], f[["m", "geo", "time", "y"]]])
    df["c"] = df["geo"].map({g: i for i, g in enumerate(EU27)})
    df["t"] = df["time"].map({y: i for i, y in enumerate(years)})
    return df.dropna(subset=["c", "t"]).reset_index(drop=True), years


def prior_mean_prec(mu0_vec, drift, sd_theta, C, T):
    n = C * T
    mean = np.empty(n)
    P = np.zeros((n, n))
    q = 1.0 / sd_theta**2
    q0 = 1.0 / SD0**2
    for c in range(C):
        base = c * T
        mean[base] = mu0_vec[c]
        for t in range(1, T):
            mean[base + t] = mean[base + t - 1] + drift
        P[base, base] += q + q0
        P[base, base + 1] -= q
        for t in range(1, T - 1):
            P[base + t, base + t] += 2 * q
            P[base + t, base + t + 1] -= q
            P[base + t, base + t - 1] -= q
        P[base + T - 1, base + T - 1] += q
        P[base + T - 1, base + T - 2] -= q
    return mean, P


def theta_posterior(phi, df, C, T):
    pm, P = prior_mean_prec(np.asarray(phi["mu0"]), float(phi["drift"]),
                            float(phi["sd_theta"]), C, T)
    BF, bF = float(phi["logB_F"]), float(phi["b_F"])
    sd_m = np.asarray(phi["sd_m"])
    Pp = P.copy()
    r = P @ pm
    for row in df.itertuples():
        idx = int(row.c) * T + int(row.t)
        if int(row.m) == 0:
            coef, off, sd = 1.0, 0.0, sd_m[0]
        else:
            coef = np.exp(BF)
            off = bF  # b_F is an offset; it is a log ratio only when B_F=1
            sd = sd_m[1]
        y = row.y - off
        w = coef / sd**2
        Pp[idx, idx] += coef * w
        r[idx] += y * w
    cov = np.linalg.inv(Pp)
    return cov @ r, cov


def make_model(df, C, T, level0, logB_loc=-0.2):
    m_arr = df["m"].to_numpy().astype(int)
    idx_arr = (df["c"] * T + df["t"]).to_numpy().astype(int)
    y_arr = df["y"].to_numpy()
    nobs = len(df)
    level0_arr = jnp.asarray(level0)

    def model():
        mu0 = numpyro.sample("mu0", dist.Normal(level0_arr, 0.75))
        drift = numpyro.sample("drift", dist.Normal(0.0, 0.10))
        log_sd_theta = numpyro.sample("log_sd_theta", dist.Normal(-1.5, 0.7))
        logB_F = numpyro.sample("logB_F", dist.Normal(logB_loc, 0.3))
        b_F = numpyro.sample("b_F", dist.Normal(0.0, 0.5))
        log_sd_m = numpyro.sample("log_sd_m", dist.Normal(jnp.array([-2.0, -1.5]),
                                                          jnp.array([0.6, 0.6])))
        sd_theta = jnp.exp(log_sd_theta)
        pm = mu0[:, None] + drift * jnp.arange(T)[None, :]
        pm = pm.reshape(-1)
        q = 1.0 / sd_theta**2
        q0 = 1.0 / SD0**2
        P = jnp.zeros((C * T, C * T))
        i0 = jnp.arange(C) * T
        P = P.at[i0, i0].add(q + q0)
        P = P.at[i0, i0 + 1].add(-q)
        inner = (jnp.arange(C)[:, None] * T + jnp.arange(1, T - 1)[None, :]).reshape(-1)
        P = P.at[inner, inner].add(2 * q)
        P = P.at[inner, inner + 1].add(-q)
        P = P.at[inner, inner - 1].add(-q)
        iN = jnp.arange(C) * T + T - 1
        P = P.at[iN, iN].add(q)
        P = P.at[iN, iN - 1].add(-q)
        cov_theta = jnp.linalg.inv(P)
        coef = jnp.where(m_arr == 0, 1.0, jnp.exp(logB_F))
        off = jnp.where(m_arr == 0, 0.0, b_F)  # b_F is an offset; log ratio only if B_F=1
        sd_obs = jnp.exp(log_sd_m)[m_arr]
        H = jnp.zeros((nobs, C * T))
        H = H.at[jnp.arange(nobs), idx_arr].set(coef)
        mean_y = H @ pm + off
        cov_y = H @ cov_theta @ H.T + jnp.diag(sd_obs**2)
        numpyro.sample("y", dist.MultivariateNormal(mean_y, covariance_matrix=cov_y),
                       obs=y_arr)

    return model


def fit(df, C, T, level0, seed=0, warm=600, samp=600, logB_loc=-0.2):
    mcmc = MCMC(NUTS(make_model(df, C, T, level0, logB_loc=logB_loc)), num_warmup=warm,
                num_samples=samp, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples()


def theta_mixture(smp, df, C, T, thin=2, seed=17001):
    take = np.arange(0, smp["mu0"].shape[0], thin)
    S = len(take)
    draws = np.empty((C * T, S))
    rng = np.random.default_rng(seed)
    for si, s in enumerate(take):
        phis = dict(mu0=np.asarray(smp["mu0"][s]), drift=float(smp["drift"][s]),
                    sd_theta=float(np.exp(smp["log_sd_theta"][s])),
                    logB_F=float(smp["logB_F"][s]), b_F=float(smp["b_F"][s]),
                    sd_m=np.exp(np.asarray(smp["log_sd_m"][s])))
        mth, cth = theta_posterior(phis, df, C, T)
        draws[:, si] = rng.multivariate_normal(mth, cth)
    return draws


def main():
    df, years = load_panel()
    C, T = len(EU27), len(years)
    print(f"panel: {len(df)} obs, {C} countries, {T} waves {years}")
    # empirical F/TOTAL ratio for reference
    tot = df[df["m"] == 0].rename(columns={"y": "y_tot"})
    f = df[df["m"] == 1].rename(columns={"y": "y_f"})
    mg = tot.merge(f, on=["c", "t"])
    ratio = np.exp(mg["y_f"] - mg["y_tot"])
    print(f"empirical F/TOTAL: median={ratio.median():.3f} "
          f"IQR=({ratio.quantile(.25):.3f},{ratio.quantile(.75):.3f})")
    # level0: per-country first-wave TOTAL_HH log value (fallback: global median)
    first_year = years[0]
    lvl = (df[(df["m"] == 0) & (df["time"] == first_year)]
           .set_index("c")["y"])
    level0 = [lvl.get(c, df[df["m"] == 0]["y"].median()) for c in range(C)]

    # ---------------- E1: full fit
    print("\n[E1] fitting full panel ...", flush=True)
    smp = fit(df, C, T, level0, seed=7)
    BF = np.exp(np.asarray(smp["logB_F"]))
    bF = np.asarray(smp["b_F"])
    print(f"[E1] B_F slope posterior: mean={BF.mean():.3f} "
          f"90%CI=({np.quantile(BF,.05):.3f},{np.quantile(BF,.95):.3f})")
    print(f"[E1] b_F offset posterior: mean={bF.mean():.3f} "
          f"90%CI=({np.quantile(bF,.05):.3f},{np.quantile(bF,.95):.3f})")
    theta_ref = float(np.median(mg["y_tot"]))
    ratio_at_ref = np.exp((BF - 1.0) * theta_ref + bF)
    print(f"[E1] implied F/TOTAL level ratio at median observed anchor theta={theta_ref:.3f}: "
          f"median={np.median(ratio_at_ref):.3f} "
          f"90%CI=({np.quantile(ratio_at_ref,.05):.3f},{np.quantile(ratio_at_ref,.95):.3f}); "
          f"empirical median={ratio.median():.3f}")
    np.save(RES / "e1_samples_logBF.npy", np.asarray(smp["logB_F"]))

    # theta mixture for hierarchy check
    draws = theta_mixture(smp, df, C, T, thin=3)
    theta_mean = draws.mean(axis=1).reshape(C, T)

    # ---------------- E3: hierarchy check vs EU27 aggregate
    agg = pd.read_csv(DATA / "env_wasgen_W121_TOTAL_HH.csv")
    agg = agg[agg["geo"] == "EU27_2020"].set_index("time")["value"]
    rows = []
    for ti, yr in enumerate(years):
        s_draws = np.exp(draws.reshape(C, T, -1)[:, ti, :]).sum(axis=0)
        lo, hi = np.quantile(s_draws, [0.05, 0.95])
        est = np.median(s_draws)
        tru = agg.get(yr, np.nan) if yr in agg.index else np.nan
        rows.append(dict(year=yr, est_Mt=est / 1e6, lo=lo / 1e6, hi=hi / 1e6,
                         truth_Mt=tru / 1e6,
                         covered=bool(lo <= tru <= hi) if tru == tru else None))
    e3 = pd.DataFrame(rows)
    e3.to_csv(RES / "e3_hierarchy_check.csv", index=False)
    print("\n[E3] hierarchy check (sum of members vs official EU27 aggregate):")
    print(e3.round(2).to_string(index=False))

    # ---------------- E2: mask-recovery for IE and DE, last two waves
    for geo in ["IE", "DE"]:
        c = EU27.index(geo)
        mask = (df["c"] == c) & (df["t"] >= T - 2)
        df_sub = df[~mask].reset_index(drop=True)
        held = df[mask]
        print(f"\n[E2] masking {geo} waves {years[-2]}, {years[-1]} "
              f"({mask.sum()} obs held out) ...", flush=True)
        smp2 = fit(df_sub, C, T, level0, seed=8)
        draws2 = theta_mixture(smp2, df_sub, C, T, thin=3)
        out = []
        for ti in [T - 2, T - 1]:
            d = np.exp(draws2[c * T + ti, :])
            lo, med, hi = np.quantile(d, [0.05, 0.5, 0.95])
            tru = float(np.exp(held[held["t"] == ti]["y"]).iloc[0])
            # persistence baseline: last observed TOTAL_HH of this country
            history = df_sub[(df_sub["c"] == c) & (df_sub["m"] == 0)].sort_values("time")
            last_obs = history["y"].iloc[-1]
            pers = np.exp(last_obs)
            out.append(dict(geo=geo, year=years[ti], truth_Mt=tru / 1e6,
                            pred_Mt=med / 1e6, lo=lo / 1e6, hi=hi / 1e6,
                            covered=bool(lo <= tru <= hi),
                            err_pct=(med - tru) / tru * 100,
                            persistence_err_pct=(pers - tru) / tru * 100))
        e2 = pd.DataFrame(out)
        e2.to_csv(RES / f"e2_mask_recovery_{geo}.csv", index=False)
        print(e2.round(2).to_string(index=False))

    print("\nsaved results/e1_*, e2_*, e3_*")


if __name__ == "__main__":
    main()
