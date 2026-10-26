"""W3a: China national CDW coherent estimation — collapsed Bayesian fit on real data.

Latent: theta_t = log(true national CDW generation, 万吨), t = 2000..2024.
Random walk with shared drift; priors grounded in EU panel descriptives (S1).

Observation processes (national, caliber=产生量, substream=总量):
  anchor   : meescc (2023, 2024)            B=1, b=0  (official city-aggregated)
  free     : gao_2025 (2000-2022, digitized Fig3), wang_2022, zheng_2017,
             envirunion_2017, mee_draft_2025 (wide caliber, lower-bound as point
             with wide sd -- approximation flagged in paper), cdw_alliance (2014,
             same treatment)
Outputs:
  results/china_national_posterior.csv  (year, median/lo/hi, 万吨)
  results/china_source_bias.csv         (per-source B, b, sd, level-ratio r_m)
  results/china_forecast.csv            (2025-2030 predictive)
  console compatibility checks for the two lower-bound claims.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(2)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "constant_drift"
RES.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2000, 2025))
T = len(YEARS)

ANCHOR = "meescc"
FIT_SOURCES = ["meescc", "gao_2025", "wang_2022", "zheng_2017",
               "envirunion_2017", "mee_draft_2025", "cdw_alliance"]
# map CSV source_ids to model sources
SRC_MAP = {"meescc_2024": "meescc", "meescc_2025": "meescc"}


def load_obs():
    df = pd.read_csv(ROOT / "data" / "china" / "china_observations_v01.csv")
    df["src"] = df["source_id"].replace(SRC_MAP)
    sel = df[(df["level"] == "national") & (df["caliber"] == "产生量")
             & (df["substream"] == "建筑垃圾总量")
             & (df["src"].isin(FIT_SOURCES))].copy()
    sel = sel[(sel["year"] >= YEARS[0]) & (sel["year"] <= YEARS[-1])]
    sel["y"] = np.log(sel["value_wt"])
    sel["t"] = sel["year"] - YEARS[0]
    sel["m"] = sel["src"].map({s: i for i, s in enumerate(FIT_SOURCES)})
    print(f"fit observations: {len(sel)}")
    print(sel[["src", "year", "value_wt"]].to_string(index=False))
    return sel.sort_values(["m", "t"]).reset_index(drop=True)


def prior_mean_prec(mu0, drift, sd_theta, T):
    mean = mu0 + drift * np.arange(T)
    P = np.zeros((T, T))
    q = 1.0 / sd_theta**2
    q0 = 1.0 / 0.6**2
    P[0, 0] += q + q0
    P[0, 1] -= q
    for t in range(1, T - 1):
        P[t, t] += 2 * q
        P[t, t + 1] -= q
        P[t, t - 1] -= q
    P[T - 1, T - 1] += q
    P[T - 1, T - 2] -= q
    return mean, P


def theta_posterior(phi, df, T):
    pm, P = prior_mean_prec(float(phi["mu0"]), float(phi["drift"]),
                            float(phi["sd_theta"]), T)
    Pp = P.copy()
    r = P @ pm
    for row in df.itertuples():
        m = int(row.m)
        coef = float(np.exp(phi["logB"][m]))
        off = float(phi["b"][m])
        sd = float(np.exp(phi["log_sd_m"][m]))
        idx = int(row.t)
        y = row.y - off
        w = coef / sd**2
        Pp[idx, idx] += coef * w
        r[idx] += y * w
    cov = np.linalg.inv(Pp)
    return cov @ r, cov


def make_model(df, T):
    m_arr = df["m"].to_numpy().astype(int)
    t_arr = df["t"].to_numpy().astype(int)
    y_arr = df["y"].to_numpy()
    M = len(FIT_SOURCES)
    nobs = len(df)

    def model():
        mu0 = numpyro.sample("mu0", dist.Normal(jnp.log(26279.0), 0.6))
        drift = numpyro.sample("drift", dist.Normal(0.08, 0.05))
        sd_theta = numpyro.sample("sd_theta", dist.HalfNormal(0.25))
        # anchor (m=0): B=1, b=0 fixed; others free
        logB_f = numpyro.sample("logB_f", dist.Normal(0.0, 0.25).expand([M - 1]))
        b_f = numpyro.sample("b_f", dist.Normal(0.0, 0.4).expand([M - 1]))
        log_sd_m = numpyro.sample(
            "log_sd_m",
            dist.Normal(jnp.log(jnp.array([0.05] + [0.15] * (M - 1))),
                        jnp.array([0.4] * M)).expand([M]).to_event(1))
        logB = jnp.concatenate([jnp.zeros(1), logB_f])
        b = jnp.concatenate([jnp.zeros(1), b_f])
        sd_m = jnp.exp(log_sd_m)
        pm = mu0 + drift * jnp.arange(T)
        q = 1.0 / sd_theta**2
        q0 = 1.0 / 0.6**2
        diag = jnp.full((T,), 2 * q).at[0].set(q + q0).at[T - 1].set(q)
        P = jnp.diag(diag) + jnp.diag(jnp.full(T - 1, -q), 1) \
            + jnp.diag(jnp.full(T - 1, -q), -1)
        cov_theta = jnp.linalg.inv(P)
        coef = jnp.exp(logB)[m_arr]
        off = b[m_arr]
        sd_obs = sd_m[m_arr]
        H = jnp.zeros((nobs, T)).at[jnp.arange(nobs), t_arr].set(coef)
        mean_y = H @ pm + off
        cov_y = H @ cov_theta @ H.T + jnp.diag(sd_obs**2)
        numpyro.sample("y", dist.MultivariateNormal(mean_y, covariance_matrix=cov_y),
                       obs=y_arr)
    return model


def main():
    df = load_obs()
    M = len(FIT_SOURCES)
    mcmc = MCMC(NUTS(make_model(df, T)), num_warmup=1200, num_samples=1200,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(42))
    smp = mcmc.get_samples()
    mcmc.print_summary(exclude_deterministic=True)

    # theta mixture posterior
    take = np.arange(0, smp["mu0"].shape[0], 2)
    S = len(take)
    draws = np.empty((T, S))
    logB_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["logB_f"])[take]], 1)
    b_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["b_f"])[take]], 1)
    r_m = np.empty((S, M))
    for si, s in enumerate(take):
        phis = dict(mu0=float(smp["mu0"][s]), drift=float(smp["drift"][s]),
                    sd_theta=float(smp["sd_theta"][s]),
                    logB=logB_all[si], b=b_all[si],
                    log_sd_m=np.asarray(smp["log_sd_m"][s]))
        mth, cth = theta_posterior(phis, df, T)
        draws[:, si] = np.random.multivariate_normal(mth, cth)
    theta_bar = float(np.mean(draws))
    r_m = (np.exp(logB_all) - 1.0) * theta_bar + b_all   # log-ratio at mean level

    med, lo, hi = np.quantile(draws, [0.5, 0.05, 0.95], axis=1)
    post = pd.DataFrame({"year": YEARS, "median_wt": np.exp(med),
                         "lo90_wt": np.exp(lo), "hi90_wt": np.exp(hi)})
    post[["median_wt", "lo90_wt", "hi90_wt"]] = post[
        ["median_wt", "lo90_wt", "hi90_wt"]].round(0)
    post.to_csv(RES / "china_national_posterior.csv", index=False, encoding="utf-8-sig")
    print(post.to_string(index=False))

    # source bias table
    rows = []
    for mi, s in enumerate(FIT_SOURCES):
        Br = np.exp(logB_all[:, mi])
        rows.append(dict(
            source=s, anchor=(s == ANCHOR),
            B_mean=float(Br.mean()), B_lo=float(np.quantile(Br, .05)),
            B_hi=float(np.quantile(Br, .95)),
            ratio_at_mean=float(np.exp(np.median(r_m[:, mi]))),
            ratio_lo=float(np.exp(np.quantile(r_m[:, mi], .05))),
            ratio_hi=float(np.exp(np.quantile(r_m[:, mi], .95))),
            sd_median=float(np.median(np.exp(np.asarray(smp["log_sd_m"])[:, mi])))))
    sb = pd.DataFrame(rows)
    sb.to_csv(RES / "china_source_bias.csv", index=False, encoding="utf-8-sig")
    print("\nsource bias:"); print(sb.round(3).to_string(index=False))

    # compatibility with lower-bound claims
    t2023 = YEARS.index(2023)
    t2014 = YEARS.index(2014)
    p30 = float((draws[t2023] > np.log(300000.0)).mean())
    p15 = float((draws[t2014] > np.log(150000.0)).mean())
    print(f"\nP(theta_2023 > 30亿吨) = {p30:.3f}   (wide-caliber claim)")
    print(f"P(theta_2014 > 15亿吨) = {p15:.3f}   (alliance claim)")

    # forecast 2025-2030: random-walk extrapolation per posterior draw
    dr = np.asarray(smp["drift"])[take]
    sdT = np.asarray(smp["sd_theta"])[take]
    F = 6
    fut = np.empty((F, S))
    rng = np.random.default_rng(1)
    for si in range(S):
        th = draws[T - 1, si]
        path = [th]
        for k in range(F):
            path.append(path[-1] + dr[si] + rng.normal(0, sdT[si]))
        fut[:, si] = path[1:]
    fm, flo, fhi = np.quantile(fut, [0.5, 0.05, 0.95], axis=1)
    fc = pd.DataFrame({"year": list(range(2025, 2031)),
                       "median_wt": np.exp(fm), "lo90_wt": np.exp(flo),
                       "hi90_wt": np.exp(fhi)}).round(0)
    fc.to_csv(RES / "china_forecast.csv", index=False, encoding="utf-8-sig")
    print("\nforecast:"); print(fc.to_string(index=False))
    np.save(RES / "china_theta_draws.npy", draws)
    print("\nsaved results/china_*")


if __name__ == "__main__":
    main()
