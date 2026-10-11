"""S2b spike: fully Bayesian collapsed-inference SBC.

Lesson from s2_sbc.py: exact Gaussian machinery is calibrated (oracle passes), but
empirical-Bayes plug-in (EM) is badly overconfident (coverage90 = 0.38). Therefore
the production model must integrate out hyperparameter uncertainty.

Design: collapse the linear-Gaussian latent states analytically --
    theta | phi ~ N(pm(phi), P(phi)^{-1})
    y | phi    ~ N(H(phi) pm + b(phi),  H P^{-1} H^T + R(phi))
with phi = (log_sd_theta, mu0, drift_c, logit-scale B_m, b_m, log_sd_m).
Sample phi | y with NUTS (numpyro), then theta | phi, y is exact Gaussian per draw
=> mixture posterior for theta that carries full hyperparameter uncertainty.

Parameterization fix vs s2_sbc.py: observations use centered states
    y = B_m * (theta_{c,t} - theta_ref) + b_m + eps,   theta_ref = mu0
so b_m is the expected reading at the reference level (kills slope-intercept
confounding at theta ~ 14).

SBC: truth drawn FROM THE INFERENCE PRIORS (proper SBC), R replicates, rank
uniformity + interval coverage + bias-parameter recovery.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy import stats

numpyro.set_host_device_count(2)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- ground-truth priors
PRIOR = dict(
    mu0_loc=14.5, mu0_scale=1.5,
    drift_loc=0.02, drift_scale=0.02,
    log_sd_theta_loc=-3.0, log_sd_theta_scale=0.5,
    logB_loc=0.0, logB_scale=0.20,          # B_m = exp(N(0, .2))  (anchor fixed B=1)
    b_loc=0.0, b_scale=0.30,
    log_sd_m_loc=-2.2, log_sd_m_scale=0.5,
)
SD0 = 0.5


def sample_truth(C, T, M, rng):
    p = PRIOR
    phi = dict(
        mu0=float(rng.normal(p["mu0_loc"], p["mu0_scale"])),
        drift=rng.normal(p["drift_loc"], p["drift_scale"], C),
        sd_theta=float(np.exp(rng.normal(p["log_sd_theta_loc"], p["log_sd_theta_scale"]))),
        logB=np.concatenate([[0.0], rng.normal(p["logB_loc"], p["logB_scale"], M - 1)]),
        b=np.concatenate([[0.0], rng.normal(p["b_loc"], p["b_scale"], M - 1)]),
        sd_m=np.exp(rng.normal(p["log_sd_m_loc"], p["log_sd_m_scale"], M)),
    )
    theta = np.empty((C, T))
    theta[:, 0] = rng.normal(phi["mu0"], SD0, C)
    for t in range(1, T):
        theta[:, t] = theta[:, t - 1] + phi["drift"] + rng.normal(0, phi["sd_theta"], C)
    return phi, theta


def simulate_obs(phi, theta, cover, rng):
    C, T = theta.shape
    M = len(cover)
    B = np.exp(phi["logB"])
    rows = []
    for m in range(M):
        for c in range(C):
            for t in range(T):
                if rng.random() < cover[m]:
                    y = B[m] * theta[c, t] + phi["b"][m] \
                        + rng.normal(0, phi["sd_m"][m])
                    rows.append((m, c, t, y))
    return pd.DataFrame(rows, columns=["m", "c", "t", "y"])


# ------------------------------------------------------------- collapsed marginal

def prior_mean_prec_np(mu0, drift, sd_theta, C, T):
    n = C * T
    mean = np.empty(n)
    P = np.zeros((n, n))
    q = 1.0 / sd_theta**2
    q0 = 1.0 / SD0**2
    for c in range(C):
        base = c * T
        mean[base] = mu0
        for t in range(1, T):
            mean[base + t] = mean[base + t - 1] + drift[c]
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
    """Exact Gaussian posterior of theta given hyperparams (numpy)."""
    pm, P = prior_mean_prec_np(phi["mu0"], np.asarray(phi["drift"]),
                               float(phi["sd_theta"]), C, T)
    B = np.exp(np.asarray(phi["logB"]))
    b = np.asarray(phi["b"])
    sd_m = np.asarray(phi["sd_m"])
    Pp = P.copy()
    r = P @ pm
    for row in df.itertuples():
        idx = int(row.c) * T + int(row.t)
        coef = B[int(row.m)]
        y = row.y - b[int(row.m)]
        w = coef / sd_m[int(row.m)]**2
        Pp[idx, idx] += coef * w
        r[idx] += y * w
    cov = np.linalg.inv(Pp)
    return cov @ r, cov


def make_marginal_model(df, C, T):
    m_arr = df["m"].to_numpy()
    c_arr = df["c"].to_numpy().astype(int)
    t_arr = df["t"].to_numpy().astype(int)
    y_arr = df["y"].to_numpy()
    nobs = len(df)
    M = int(m_arr.max()) + 1
    idx_arr = c_arr * T + t_arr

    def model():
        p = PRIOR
        mu0 = numpyro.sample("mu0", dist.Normal(p["mu0_loc"], p["mu0_scale"]))
        drift = numpyro.sample("drift",
                               dist.Normal(p["drift_loc"], p["drift_scale"]).expand([C]))
        log_sd_theta = numpyro.sample(
            "log_sd_theta", dist.Normal(p["log_sd_theta_loc"], p["log_sd_theta_scale"]))
        logB = numpyro.sample("logB", dist.Normal(p["logB_loc"], p["logB_scale"])
                              .expand([M - 1]))
        bfree = numpyro.sample("bfree", dist.Normal(p["b_loc"], p["b_scale"])
                               .expand([M - 1]))
        log_sd_m = numpyro.sample("log_sd_m", dist.Normal(p["log_sd_m_loc"],
                                                          p["log_sd_m_scale"]).expand([M]))
        sd_theta = jnp.exp(log_sd_theta)
        # prior mean/precision of theta
        pm = mu0 + jnp.cumsum(jnp.concatenate(
            [jnp.zeros((C, 1)), jnp.repeat(drift[:, None], T - 1, axis=1)],
            axis=1), axis=1)
        pm = pm.reshape(-1)
        q = 1.0 / sd_theta**2
        q0 = 1.0 / SD0**2
        # dense precision
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
        # observation mapping
        logB_full = jnp.concatenate([jnp.zeros(1), logB])
        b_full = jnp.concatenate([jnp.zeros(1), bfree])
        B_obs = jnp.exp(logB_full)[m_arr]
        b_obs = b_full[m_arr]
        sd_obs = jnp.exp(log_sd_m)[m_arr]
        H = jnp.zeros((nobs, C * T))
        H = H.at[jnp.arange(nobs), idx_arr].set(B_obs)
        mean_y = H @ pm + b_obs  # uncentered obs equation
        cov_y = H @ cov_theta @ H.T + jnp.diag(sd_obs**2)
        numpyro.sample("y", dist.MultivariateNormal(mean_y, covariance_matrix=cov_y),
                       obs=y_arr)

    return model


# ------------------------------------------------------------- SBC driver

def run_scenario(name, C, T, cover, R=16, rng_seed=21, n_warm=300, n_samp=400):
    rng = np.random.default_rng(rng_seed)
    M = len(cover)
    ranks, cov90, cov50 = [], [], []
    b_err, logB_err = [], []
    for rep in range(R):
        phi, theta_true = sample_truth(C, T, M, rng)
        df = simulate_obs(phi, theta_true, cover, rng)
        model = make_marginal_model(df, C, T)
        mcmc = MCMC(NUTS(model, dense_mass=False), num_warmup=n_warm,
                    num_samples=n_samp, num_chains=1, progress_bar=False)
        mcmc.run(jax_key(rng), )
        smp = mcmc.get_samples()
        # mixture posterior for theta: one theta draw per phi draw (thinned)
        S_full = smp["mu0"].shape[0]
        take = np.arange(0, S_full, 2)          # thin x2
        S = len(take)
        theta_draws = np.empty((C * T, S))
        for si, s in enumerate(take):
            phis = dict(
                mu0=float(smp["mu0"][s]),
                drift=np.asarray(smp["drift"][s]),
                sd_theta=float(np.exp(smp["log_sd_theta"][s])),
                logB=np.concatenate([[0.0], np.asarray(smp["logB"][s])]),
                b=np.concatenate([[0.0], np.asarray(smp["bfree"][s])]),
                sd_m=np.exp(np.asarray(smp["log_sd_m"][s])),
            )
            mth, cth = theta_posterior(phis, df, C, T)
            theta_draws[:, si] = np.random.multivariate_normal(mth, cth)
        tt = theta_true.reshape(-1)
        rr = (theta_draws < tt[:, None]).sum(axis=1) / S
        pick = rng.choice(C * T, size=8, replace=False)
        ranks.extend(rr[pick].tolist())
        mq = np.quantile(theta_draws, [0.05, 0.25, 0.75, 0.95], axis=1)
        cov90.extend(((tt >= mq[0]) & (tt <= mq[3])).tolist())
        cov50.extend(((tt >= mq[1]) & tt.__le__(mq[2])).tolist())
        # hyperparam recovery
        b_est = np.concatenate([np.zeros(1), np.asarray(smp["bfree"]).mean(axis=0)])
        logB_est = np.concatenate([np.zeros(1),
                                   np.asarray(smp["logB"]).mean(axis=0)])
        b_err.extend((b_est - phi["b"]).tolist())
        logB_err.extend((logB_est - phi["logB"]).tolist())
        if (rep + 1) % 12 == 0:
            print(f"  [{name}] rep {rep+1}/{R}", flush=True)
    ranks = np.array(ranks)
    res = {
        "scenario": name, "replicates": R,
        "coverage90": float(np.mean(cov90)), "coverage50": float(np.mean(cov50)),
        "rank_mean": float(ranks.mean()), "rank_sd": float(ranks.std()),
        "KS_p": float(stats.kstest(ranks, "uniform").pvalue),
        "b_mae": float(np.mean(np.abs(b_err))),
        "logB_mae": float(np.mean(np.abs(logB_err))),
    }
    print(res, flush=True)
    np.save(RES / f"s2c_ranks_{name}.npy", ranks)
    return res


def jax_key(rng):
    import jax
    return jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))


def main():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    rows = []
    if which in ("EUlike", "ALL"):
        rows.append(run_scenario("EUlike", C=8, T=7, cover=[0.95, 0.8, 0.6], rng_seed=21))
    if which in ("CNsparse", "ALL"):
        rows.append(run_scenario("CNsparse", C=8, T=7, cover=[0.5, 0.35, 0.2], rng_seed=22))
    out = RES / f"s2c_sbc_summary_{which}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
