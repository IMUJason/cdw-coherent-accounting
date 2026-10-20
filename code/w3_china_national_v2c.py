"""W3a-v2: China national model with mean-reverting growth (audit-driven fix).

Audit trail: v1 (constant-drift random walk) forecast overshoots a visibly
saturating series (2030 median 41.2亿吨). Root cause: constant drift is
incompatible with post-2019 deceleration (real-estate downturn + renewal-shift).
Fix: increments g_t = theta_t - theta_{t-1} follow AR(1) mean reversion to a
long-run growth c,  g_t = c + rho*(g_{t-1} - c) + eta  -> growth decays
geometrically to c; still linear-Gaussian, so collapsed inference unchanged
(precision becomes D' Q_z D with AR(1) precision on increments).

Outputs overwrite results/china_* (v2 suffix on summary print).
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
RES = ROOT / "results"
YEARS = list(range(2000, 2025))
T = len(YEARS)

ANCHOR = "meescc"
FIT_SOURCES = ["meescc", "gao_2025", "wang_2022", "zheng_2017",
               "envirunion_2017", "mee_draft_2025", "cdw_alliance"]
SRC_MAP = {"meescc_2024": "meescc", "meescc_2025": "meescc"}
SD0 = 0.6


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
    return sel.sort_values(["m", "t"]).reset_index(drop=True)


def ar1_prec(Tm1, rho, sd_eta):
    """Precision of g_1..g_Tm1 ~ stationary AR(1) with innovation sd_eta.
    Q[1,1] = (1-rho^2)/s^2 + rho^2/s^2 = q  (stationary-start factor is exactly
    cancelled by the rho^2 contribution of the next transition);
    interior q(1+rho^2); endpoint q; off-diagonal -q*rho."""
    q = 1.0 / sd_eta**2
    Q = np.zeros((Tm1, Tm1))
    if Tm1 == 0:
        return Q
    if Tm1 == 1:
        Q[0, 0] = q * (1 - rho**2)
        return Q
    Q[0, 0] = q
    for i in range(1, Tm1):
        Q[i, i] += q * (1 + rho**2) if i < Tm1 - 1 else q
        Q[i, i - 1] -= q * rho
        Q[i - 1, i] -= q * rho
    return Q


def theta_prior(mu0, c, rho, sd_eta, T):
    """Mean/precision of theta under increment AR(1). z = (theta_0, g_1..g_{T-1}),
    theta = G z with G lower-triangular ones; P_theta = D' Q_z D, D = G^{-1}."""
    D = np.eye(T)
    D[np.arange(1, T), np.arange(0, T - 1)] = -1.0   # D theta = (theta_0, g)
    Qz = np.zeros((T, T))
    Qz[0, 0] = 1.0 / SD0**2
    Qz[1:, 1:] = ar1_prec(T - 1, rho, sd_eta)
    P = D.T @ Qz @ D
    # mean: theta_0 ~ mu0; g_t mean = c (stationary mean)
    mz = np.concatenate([[mu0], np.full(T - 1, c)])
    mean = np.linalg.solve(D, mz)   # theta = G z = D^{-1} z
    return mean, P


def theta_posterior(phi, df, T):
    pm, P = theta_prior(float(phi["mu0"]), float(phi["c"]), float(phi["rho"]),
                        float(phi["sd_eta"]), T)
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


def make_model(df, T, *, gao_corr=0.0, unit_slopes=False, mu0_scale=0.6,
               logB_scale=0.25, offset_scale=0.4):
    m_arr = df["m"].to_numpy().astype(int)
    t_arr = df["t"].to_numpy().astype(int)
    y_arr = df["y"].to_numpy()
    M = len(FIT_SOURCES)
    nobs = len(df)

    def model():
        mu0 = numpyro.sample("mu0", dist.Normal(jnp.log(26279.0), mu0_scale))
        c = numpyro.sample("c", dist.Normal(0.02, 0.03))
        rho_raw = numpyro.sample("rho_raw", dist.Normal(0.0, 1.0))
        rho = jax.nn.sigmoid(rho_raw) * 0.7 + 0.28   # in (0.28, 0.98)
        sd_eta = numpyro.sample("sd_eta", dist.HalfNormal(0.12))
        if unit_slopes:
            logB_f = numpyro.deterministic("logB_f", jnp.zeros(M - 1))
        else:
            logB_f = numpyro.sample("logB_f", dist.Normal(0.0, logB_scale).expand([M - 1]))
        b_f = numpyro.sample("b_f", dist.Normal(0.0, offset_scale).expand([M - 1]))
        log_sd_m_free = numpyro.sample(
            "log_sd_m_free",
            dist.Normal(jnp.log(jnp.array([0.05] + [0.15] * (M - 2))),
                        0.4).expand([M - 1]))
        # sigma for the digitized WGR series (m=1) is FIXED at 0.07:
        # +-5% digitization error (log-sd .05) plus model discrepancy (.05)
        log_sd_m = jnp.concatenate(
            [log_sd_m_free[:1], jnp.array([jnp.log(0.07)]), log_sd_m_free[1:]])
        logB = jnp.concatenate([jnp.zeros(1), logB_f])
        b = jnp.concatenate([jnp.zeros(1), b_f])
        sd_m = jnp.exp(log_sd_m)
        # prior mean/precision of theta (jax version of theta_prior)
        Dz = jnp.eye(T).at[jnp.arange(1, T), jnp.arange(0, T - 1)].set(-1.0)
        q = 1.0 / sd_eta**2
        Qz = jnp.zeros((T, T))
        Qz = Qz.at[0, 0].set(1.0 / SD0**2)
        # g-block (rows/cols 1..T-1): AR(1) precision
        gidx = jnp.arange(1, T)
        Qz = Qz.at[gidx, gidx].set(q * (1 + rho**2))
        Qz = Qz.at[1, 1].set(q)                      # first diag = q (stationary
        Qz = Qz.at[T - 1, T - 1].set(q)              # start cancels; endpoint = q)
        off = jnp.arange(2, T)                        # g-internal couplings only
        Qz = Qz.at[off, off - 1].set(-q * rho)
        Qz = Qz.at[off - 1, off].set(-q * rho)
        P = Dz.T @ Qz @ Dz
        mz = jnp.concatenate([jnp.array([mu0]), jnp.full(T - 1, c)])
        pm = jnp.linalg.solve(Dz, mz)
        cov_theta = jnp.linalg.inv(P)
        coef = jnp.exp(logB)[m_arr]
        off = b[m_arr]
        sd_obs = sd_m[m_arr]
        H = jnp.zeros((nobs, T)).at[jnp.arange(nobs), t_arr].set(coef)
        mean_y = H @ pm + off
        R_obs = jnp.diag(sd_obs**2)
        # Optional sensitivity: serial observation error in the digitized series.
        # The default remains the archived v2c likelihood (independent errors).
        if gao_corr:
            same_gao = (m_arr[:, None] == 1) & (m_arr[None, :] == 1)
            lag = np.abs(t_arr[:, None] - t_arr[None, :])
            corr_extra = np.where(same_gao & (lag > 0), gao_corr**lag, 0.0)
            R_obs = R_obs + sd_obs[:, None] * sd_obs[None, :] * jnp.asarray(corr_extra)
        cov_y = H @ cov_theta @ H.T + R_obs
        numpyro.sample("y", dist.MultivariateNormal(mean_y, covariance_matrix=cov_y),
                       obs=y_arr)
        numpyro.deterministic("rho", rho)
    return model


def main(mu0_scale=0.6, seed=45, out_suffix="v2c", state_seed=45001):
    df = load_obs()
    M = len(FIT_SOURCES)
    tag = f"_{out_suffix}" if out_suffix else ""
    mcmc = MCMC(NUTS(make_model(df, T, mu0_scale=mu0_scale)), num_warmup=1500, num_samples=1500,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    smp = mcmc.get_samples()
    mcmc.print_summary(exclude_deterministic=False)

    take = np.arange(0, smp["mu0"].shape[0], 2)
    S = len(take)
    draws = np.empty((T, S))
    logB_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["logB_f"])[take]], 1)
    b_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["b_f"])[take]], 1)
    lsd = np.asarray(smp["log_sd_m_free"])
    state_rng = np.random.default_rng(state_seed)
    for si, s in enumerate(take):
        phis = dict(mu0=float(smp["mu0"][s]), c=float(smp["c"][s]),
                    rho=float(smp["rho"][s]), sd_eta=float(smp["sd_eta"][s]),
                    logB=logB_all[si], b=b_all[si],
                    log_sd_m=np.concatenate([lsd[s][:1], [np.log(0.07)], lsd[s][1:]]))
        mth, cth = theta_posterior(phis, df, T)
        draws[:, si] = state_rng.multivariate_normal(mth, cth)
    theta_bar = float(np.mean(draws))
    r_m = (np.exp(logB_all) - 1.0) * theta_bar + b_all

    med, lo, hi = np.quantile(draws, [0.5, 0.05, 0.95], axis=1)
    post = pd.DataFrame({"year": YEARS, "median_wt": np.exp(med),
                         "lo90_wt": np.exp(lo), "hi90_wt": np.exp(hi)}).round(0)
    post.to_csv(RES / f"china_national_posterior{tag}.csv", index=False,
                encoding="utf-8-sig")
    print(post.to_string(index=False))

    rows = []
    for mi, s in enumerate(FIT_SOURCES):
        Br = np.exp(logB_all[:, mi])
        rows.append(dict(
            source=s, anchor=(s == ANCHOR),
            ratio_at_mean=float(np.exp(np.median(r_m[:, mi]))),
            ratio_lo=float(np.exp(np.quantile(r_m[:, mi], .05))),
            ratio_hi=float(np.exp(np.quantile(r_m[:, mi], .95))),
            sd_median=(0.07 if mi == 1 else
                       float(np.median(np.exp(np.asarray(smp["log_sd_m_free"])[:, mi - 1])))
                       if mi > 1 else
                       float(np.median(np.exp(np.asarray(smp["log_sd_m_free"])[:, 0]))))))
    sb = pd.DataFrame(rows)
    sb.to_csv(RES / f"china_source_bias{tag}.csv", index=False, encoding="utf-8-sig")
    print("\nsource bias (v2):"); print(sb.round(3).to_string(index=False))

    # forecast with mean reversion: g decays to c at rate rho
    t2023, t2014 = YEARS.index(2023), YEARS.index(2014)
    print(f"\nP(theta_2023 > 30亿吨) = {(draws[t2023] > np.log(300000.)).mean():.3f}")
    print(f"P(theta_2014 > 15亿吨) = {(draws[t2014] > np.log(150000.)).mean():.3f}")

    F = 6
    R = 50  # simulated continuations per posterior draw
    fut = np.empty((F, S * R))
    rng = np.random.default_rng(1)
    for si, s in enumerate(take):
        c, rho, sd_eta = float(smp["c"][s]), float(smp["rho"][s]), float(smp["sd_eta"][s])
        g0 = draws[T - 1, si] - draws[T - 2, si]
        th0 = draws[T - 1, si]
        for r in range(R):
            g, th = g0, th0
            for k in range(F):
                g = c + rho * (g - c) + rng.normal(0, sd_eta)
                th = th + g
                fut[k, si * R + r] = th
    fm, flo, fhi = np.quantile(fut, [0.5, 0.05, 0.95], axis=1)
    fc = pd.DataFrame({"year": list(range(2025, 2031)), "median_wt": np.exp(fm),
                       "lo90_wt": np.exp(flo), "hi90_wt": np.exp(fhi)}).round(0)
    fc.to_csv(RES / f"china_forecast{tag}.csv", index=False, encoding="utf-8-sig")
    print("\nforecast v2 (mean-reverting):"); print(fc.to_string(index=False))
    np.save(RES / f"china_theta_draws{tag}.npy", draws)
    np.save(RES / f"china_hyper_draws{tag}.npy",
            np.column_stack([np.asarray(smp["c"])[take],
                             np.asarray(smp["rho"])[take],
                             np.asarray(smp["sd_eta"])[take]]))
    np.save(RES / f"china_forecast_draws{tag}.npy", np.exp(fut))
    print(f"\nrho posterior: mean={float(np.mean(smp['rho'])):.3f}  "
          f"c (long-run growth): mean={float(np.mean(smp['c'])):.4f}")
    print("saved results/china_*_v2.*")


if __name__ == "__main__":
    main()
