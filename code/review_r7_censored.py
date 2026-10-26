"""One-sided censored-likelihood refit for the two lower-bound observations.

The MEE draft statement ("> 3.0 Gt", 2023) and the 2014 industry alliance
report ("> 1.5 Gt") are published as lower bounds. The production model
enters them as wide-error point observations at the bound. This diagnostic
replaces those two likelihood terms with P(y >= bound) under the same
collapsed marginal Gaussian, keeping every other model setting identical.

State draws use importance weighting: theta is drawn from the exact-data
Gaussian conditional and reweighted by the censored probabilities given
theta. Outputs are isolated under results/review_r7/censored/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS
import pandas as pd
from jax.scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import w3_china_national_v2c as v2c  # noqa: E402

OUT = ROOT / "results" / "review_r7" / "censored"
SEED = 71001
STATE_SEED = 71002
WARMUP = 1500
SAMPLES = 1500
CENSORED_SOURCES = {"mee_draft_2025", "cdw_alliance"}
TARGET_ACCEPT = 0.9


def build_model(df_exact: pd.DataFrame, df_cens: pd.DataFrame, T: int):
    m_arr = df_exact["m"].to_numpy(dtype=int)
    t_arr = df_exact["t"].to_numpy(dtype=int)
    y_arr = df_exact["y"].to_numpy(dtype=float)
    nobs = len(df_exact)
    cm = df_cens["m"].to_numpy(dtype=int)
    ct = df_cens["t"].to_numpy(dtype=int)
    cbound = df_cens["y"].to_numpy(dtype=float)
    ncens = len(df_cens)
    M = len(v2c.FIT_SOURCES)

    def model():
        mu0 = numpyro.sample("mu0", dist.Normal(jnp.log(26279.0), 0.6))
        c = numpyro.sample("c", dist.Normal(0.02, 0.03))
        rho_raw = numpyro.sample("rho_raw", dist.Normal(0.0, 1.0))
        rho = jax.nn.sigmoid(rho_raw) * 0.7 + 0.28
        sd_eta = numpyro.sample("sd_eta", dist.HalfNormal(0.12))
        numpyro.deterministic("rho", rho)
        logB_f = numpyro.sample("logB_f", dist.Normal(0.0, 0.25).expand([M - 1]))
        b_f = numpyro.sample("b_f", dist.Normal(0.0, 0.4).expand([M - 1]))
        log_sd_m_free = numpyro.sample(
            "log_sd_m_free",
            dist.Normal(jnp.log(jnp.array([0.05] + [0.15] * (M - 2))), 0.4).expand([M - 1]))
        log_sd_m = jnp.concatenate(
            [log_sd_m_free[:1], jnp.array([jnp.log(0.07)]), log_sd_m_free[1:]])
        logB = jnp.concatenate([jnp.zeros(1), logB_f])
        b = jnp.concatenate([jnp.zeros(1), b_f])
        sd_m = jnp.exp(log_sd_m)

        Dz = jnp.eye(T).at[jnp.arange(1, T), jnp.arange(0, T - 1)].set(-1.0)
        q = 1.0 / sd_eta ** 2
        Qz = jnp.zeros((T, T))
        Qz = Qz.at[0, 0].set(1.0 / v2c.SD0 ** 2)
        gidx = jnp.arange(1, T)
        Qz = Qz.at[gidx, gidx].set(q * (1 + rho ** 2))
        Qz = Qz.at[1, 1].set(q)
        Qz = Qz.at[T - 1, T - 1].set(q)
        off = jnp.arange(2, T)
        Qz = Qz.at[off, off - 1].set(-q * rho)
        Qz = Qz.at[off - 1, off].set(-q * rho)
        P = Dz.T @ Qz @ Dz
        mz = jnp.concatenate([jnp.array([mu0]), jnp.full(T - 1, c)])
        pm = jnp.linalg.solve(Dz, mz)
        cov_theta = jnp.linalg.inv(P)

        coef = jnp.exp(logB)
        H = jnp.zeros((nobs, T)).at[jnp.arange(nobs), t_arr].set(coef[m_arr])
        mean_y = H @ pm + b[m_arr]
        cov_y = H @ cov_theta @ H.T + jnp.diag(sd_m[m_arr] ** 2)
        numpyro.sample("y_exact", dist.MultivariateNormal(mean_y, covariance_matrix=cov_y),
                       obs=y_arr)

        # Censoring conditions on the exact data: with (y_exact, y_c) jointly
        # Gaussian given phi, P(y_c >= bound | y_exact, phi) is a univariate
        # Gaussian tail with the exact-data conditional mean and variance.
        m_c0 = coef[cm] * pm[ct] + b[cm]
        H_c = jnp.zeros((ncens, T)).at[jnp.arange(ncens), ct].set(coef[cm])
        cross = H_c @ cov_theta @ H.T
        resid = y_arr - mean_y
        solve_ee = jnp.linalg.solve(cov_y, cross.T)
        m_cond = m_c0 + cross @ jnp.linalg.solve(cov_y, resid)
        v_c0 = (coef[cm] ** 2) * jnp.diag(cov_theta)[ct] + sd_m[cm] ** 2
        v_cond = v_c0 - jnp.diag(cross @ solve_ee)
        # The two bounds enter as independent conditional one-dimensional
        # tails; their conditional cross-covariance is neglected.
        numpyro.factor("censored_bounds",
                       jnp.sum(log_ndtr((m_cond - jnp.asarray(cbound)) / jnp.sqrt(v_cond))))

    return model


def main() -> None:
    started = time.monotonic()
    df = v2c.load_obs()
    cens_mask = df["src"].isin(CENSORED_SOURCES)
    if int(cens_mask.sum()) != 2:
        raise ValueError(f"Expected exactly 2 lower-bound rows, found {int(cens_mask.sum())}")
    df_cens = df[cens_mask].reset_index(drop=True)
    df_exact = df[~cens_mask].reset_index(drop=True)
    print(f"exact obs={len(df_exact)}, censored={len(df_cens)}: "
          f"{df_cens['src'].tolist()}", flush=True)

    mcmc = MCMC(NUTS(build_model(df_exact, df_cens, v2c.T), target_accept_prob=TARGET_ACCEPT),
                num_warmup=WARMUP, num_samples=SAMPLES, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(SEED), extra_fields=("diverging",))
    smp = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    take = np.arange(0, SAMPLES, 2)
    S = len(take)
    logB_all = np.concatenate([np.zeros((S, 1)), smp["logB_f"][take]], axis=1)
    b_all = np.concatenate([np.zeros((S, 1)), smp["b_f"][take]], axis=1)
    lsd = smp["log_sd_m_free"]

    theta = np.empty((v2c.T, S))
    rng = np.random.default_rng(STATE_SEED)
    log_w = np.empty(S)
    cm = df_cens["m"].to_numpy(dtype=int)
    ct = df_cens["t"].to_numpy(dtype=int)
    cbound = df_cens["y"].to_numpy(dtype=float)
    from scipy.special import ndtr
    for j, s in enumerate(take):
        phi = dict(mu0=float(smp["mu0"][s]), c=float(smp["c"][s]),
                   rho=float(smp["rho"][s]), sd_eta=float(smp["sd_eta"][s]),
                   logB=logB_all[j], b=b_all[j],
                   log_sd_m=np.concatenate([lsd[s, :1], [np.log(0.07)], lsd[s, 1:]]))
        mean, cov = v2c.theta_posterior(phi, df_exact, v2c.T)
        theta[:, j] = rng.multivariate_normal(mean, cov)
        probs = ndtr((np.exp(phi["logB"])[cm] * theta[ct, j] + phi["b"][cm] - cbound)
                     / np.exp(phi["log_sd_m"])[cm])
        log_w[j] = np.sum(np.log(np.clip(probs, 1e-300, None)))
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)

    def wq(values, probs_weights):
        order = np.argsort(values)
        cum = np.cumsum(probs_weights[order])
        return np.interp([0.05, 0.5, 0.95], cum, values[order])

    gt = np.exp(theta) / 1e5
    rows = []
    for year in (2023, 2024):
        q = wq(gt[v2c.YEARS.index(year)], w)
        rows.append(dict(scope="posterior", year=year, p05=q[0], median=q[1], p95=q[2]))
    horizon = 6
    fut = np.empty((horizon, S))
    for j in range(S):
        c_ = float(smp["c"][take[j]])
        rho_ = float(smp["rho"][take[j]])
        sd_ = float(smp["sd_eta"][take[j]])
        g = theta[-1, j] - theta[-2, j]
        th = theta[-1, j]
        for k in range(horizon):
            g = c_ + rho_ * (g - c_) + rng.normal(0, sd_)
            th = th + g
            fut[k, j] = th
    fgt = np.exp(fut) / 1e5
    for year in (2025, 2030):
        q = wq(fgt[year - 2025], w)
        rows.append(dict(scope="forecast", year=year, p05=q[0], median=q[1], p95=q[2]))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "censored_summary.csv", index=False)

    diag = numpyro_summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
    rhats = np.concatenate([np.atleast_1d(v["r_hat"]).ravel() for v in diag.values()])
    esses = np.concatenate([np.atleast_1d(v["n_eff"]).ravel() for v in diag.values()])
    meta = {
        "purpose": "censored-likelihood diagnostic for the two lower-bound observations",
        "censored_sources": df_cens["src"].tolist(),
        "exact_observations": int(len(df_exact)),
        "seed": SEED, "state_seed": STATE_SEED,
        "warmup": WARMUP, "retained_draws": SAMPLES, "thinned_state_draws": int(S),
        "target_accept": TARGET_ACCEPT, "float64": bool(jax.config.x64_enabled),
        "diagnostics": {"divergences": int(np.asarray(
            mcmc.get_extra_fields()["diverging"]).sum()),
            "max_r_hat": float(np.nanmax(rhats)), "min_ess": float(np.nanmin(esses))},
        "importance": {"log_weight_range": [float(log_w.min()), float(log_w.max())],
                        "effective_sample_size": float(ess)},
        "runtime_seconds": float(time.monotonic() - started),
        "jax_version": jax.__version__, "numpyro_version": numpyro.__version__,
    }
    (OUT / "censored_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.round(3).to_string(index=False))
    print(json.dumps(meta["diagnostics"]), "IS-ESS", round(ess, 1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
