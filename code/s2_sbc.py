"""S2 spike: simulation-based calibration (SBC) for the latent CDW state-space model.

Model (log scale):
  theta_{c,t} = theta_{c,t-1} + drift_c + eta,  eta ~ N(0, sd_theta^2)   (random walk w/ drift)
  theta_{c,1} ~ N(mu0, sd0^2)
  obs i from source m on region c at time t:
      y_i = B_m * theta_{c,t} + b_m + eps,  eps ~ N(0, sd_m^2)
Plus an aggregate observation process on the sum of regions (exp-sum in levels is
approximated on the log scale by observing logSumExp via a linearization-free trick:
we observe the total in *levels* with lognormal error, implemented as
log(sum_c exp(theta)) + eps_agg; the Jacobian is used in the E-step). For the spike
we keep the aggregate observation optional and linearize it at the posterior mean
(Gauss-Newton, converges in a few iterations for near-linear cases).

Identification anchor: source m=1 is assumed unbiased on average (b_1 = 0 fixed,
B_1 = 1 fixed) -- this is THE identification assumption to be stated in the paper.

Everything is linear-Gaussian conditional on (B, b, sd_m, sd_theta), so the exact
posterior of theta is multivariate normal via precision algebra (dense; spike sizes
are small). Unknown bias/noise params are estimated by EM (empirical Bayes).

SBC protocol: R replicates; per replicate sample ground truth from the prior,
simulate observations, fit (oracle or EM), draw posterior samples, record rank of
true theta components + coverage of nominal intervals + bias-parameter recovery.
Uniform ranks => calibrated posterior.

Two scenarios:
  EU-like   : C=8 regions, T=7 waves, 3 sources with rich overlap (~80% cells)
  CN-sparse : C=8 regions, T=7 waves, 3 sources each covering 3-5 waves,
              some regions seen by only 1-2 sources (~35% cells)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import solve
from scipy import stats

RNG = np.random.default_rng(20261009)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- model utils

def build_prior(C, T, drift, sd_theta, mu0, sd0):
    """Prior mean and precision for stacked theta (C*T,). Random walk per region."""
    n = C * T
    mean = np.empty(n)
    P = np.zeros((n, n))
    for c in range(C):
        base = c * T
        mean[base] = mu0
        for t in range(1, T):
            mean[base + t] = mean[base + t - 1] + drift[c]
        # random-walk precision: tridiagonal, theta_1 prior sd0
        q = 1.0 / sd_theta**2
        q0 = 1.0 / sd0**2
        P[base, base] += q + q0
        P[base, base + 1] -= q
        for t in range(1, T - 1):
            P[base + t, base + t] += 2 * q
            P[base + t, base + t + 1] -= q
            P[base + t, base + t - 1] -= q
        P[base + T - 1, base + T - 1] += q
        P[base + T - 1, base + T - 2] -= q
    return mean, P


def posterior(theta_prior_mean, P_prior, obs, C, T):
    """Exact Gaussian posterior given obs list of (c, t, coef, y, sd).
    obs rows are linear: y = coef * theta_{c,t} + eps, eps ~ N(0, sd^2).
    (bias b_m must be pre-subtracted from y by caller)"""
    P = P_prior.copy()
    r = P_prior @ theta_prior_mean
    for c, t, coef, y, sd in obs:
        idx = c * T + t
        w = coef / sd**2
        P[idx, idx] += coef * w
        r[idx] += y * w
    cov = np.linalg.inv(P)
    mean = cov @ r
    return mean, cov


def simulate(C, T, M, drift, sd_theta, mu0, sd0, B, b, sd_m, cover, rng):
    """Sample truth + observations. cover: (M,) per-source cell coverage prob."""
    theta = np.empty((C, T))
    theta[:, 0] = rng.normal(mu0, sd0, C)
    for t in range(1, T):
        theta[:, t] = theta[:, t - 1] + drift + rng.normal(0, sd_theta, C)
    rows = []
    for m in range(M):
        for c in range(C):
            for t in range(T):
                if rng.random() < cover[m]:
                    y = B[m] * theta[c, t] + b[m] + rng.normal(0, sd_m[m])
                    rows.append((m, c, t, y))
    return theta, pd.DataFrame(rows, columns=["m", "c", "t", "y"])


# ---------------------------------------------------------------- EM fit

def em_fit(df, C, T, drift, sd_theta, mu0, sd0, anchor=(0,), n_iter=60, tol=1e-7):
    """Estimate B, b, sd_m per source by EM; anchor sources fixed at B=1,b=0."""
    M = df["m"].nunique()
    srcs = sorted(df["m"].unique())
    B = np.array([1.0] * M)
    b = np.zeros(M)
    sd_m = np.array([df.groupby("m")["y"].std().fillna(1.0).reindex(srcs).to_numpy()[i]
                     for i in range(M)])
    pm, P0 = build_prior(C, T, drift, sd_theta, mu0, sd0)
    last_ll = -np.inf
    for it in range(n_iter):
        obs = [(int(r.c), int(r.t), B[r.m], r.y - b[r.m], sd_m[r.m])
               for r in df.itertuples()]
        mean, cov = posterior(pm, P0, obs, C, T)
        # M-step per source: WLS of y on theta with posterior moments
        for mi, m in enumerate(srcs):
            sub = df[df["m"] == m]
            idx = (sub["c"] * T + sub["t"]).to_numpy()
            Eth = mean[idx]
            Eth2 = Eth**2 + cov[idx, idx]
            y = sub["y"].to_numpy()
            if m in anchor:
                resid = y - Eth
                b[mi] = 0.0
                B[mi] = 1.0
                sd_m[mi] = np.sqrt(np.mean(resid**2 + cov[idx, idx]))
            else:
                A = np.array([[Eth2.sum(), Eth.sum()], [Eth.sum(), len(sub)]])
                rhs = np.array([(y * Eth).sum(), y.sum()])
                Bm, bm = solve(A, rhs)
                B[mi], b[mi] = Bm, bm
                resid = y - (Bm * Eth + bm)
                sd_m[mi] = np.sqrt(np.mean(resid**2 + Bm**2 * cov[idx, idx]))
        # marginal loglik (approx via complete-data surrogate for convergence watch)
        obs_full = [(int(r.c), int(r.t), B[r.m], r.y - b[r.m], sd_m[r.m])
                    for r in df.itertuples()]
        mean2, _ = posterior(pm, P0, obs_full, C, T)
        ll = -float(np.sum([( (r.y - b[r.m] - B[r.m]*mean2[int(r.c)*T+int(r.t)])**2 )
                            / (2*sd_m[r.m]**2) + np.log(sd_m[r.m])
                            for r in df.itertuples()]))
        if abs(ll - last_ll) < tol * max(1.0, abs(ll)):
            break
        last_ll = ll
    return B, b, sd_m, mean, cov, it + 1


# ---------------------------------------------------------------- SBC driver

def run_scenario(name, C, T, cover, R=120, use_em=True, seed=1):
    rng = np.random.default_rng(seed)
    M = len(cover)
    # priors for ground-truth params
    out_ranks, out_cov90, out_cov50 = [], [], []
    bhat_err, Bhat_err = [], []
    for rep in range(R):
        drift = rng.normal(0.02, 0.01, C)          # ~2%/wave growth in log scale
        sd_theta = float(rng.uniform(0.03, 0.08))
        mu0, sd0 = float(rng.uniform(13.5, 15.5)), 0.5   # ~ Mt scale in log tonnes
        B_true = np.array([1.0] + list(rng.uniform(0.6, 1.3, M - 1)))
        b_true = np.array([0.0] + list(rng.normal(0, 0.25, M - 1)))
        sd_true = rng.uniform(0.03, 0.15, M)
        theta_true, df = simulate(C, T, M, drift, sd_theta, mu0, sd0,
                                  B_true, b_true, sd_true, cover, rng)
        pm, P0 = build_prior(C, T, drift, sd_theta, mu0, sd0)
        if use_em:
            B_e, b_e, sd_e, mean, cov, _ = em_fit(
                df, C, T, drift, sd_theta, mu0, sd0, anchor=(0,))
        else:
            B_e, b_e, sd_e = B_true, b_true, sd_true
        obs = [(int(r.c), int(r.t), B_e[r.m], r.y - b_e[r.m], sd_e[r.m])
               for r in df.itertuples()]
        mean, cov = posterior(pm, P0, obs, C, T)
        sd_post = np.sqrt(np.diag(cov))
        # rank statistics via posterior samples
        S = 400
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(C * T))
        draws = mean[:, None] + L @ rng.standard_normal((C * T, S))
        tt = theta_true.reshape(-1)
        ranks = (draws < tt[:, None]).sum(axis=1) / S
        pick = rng.choice(C * T, size=8, replace=False)
        out_ranks.extend(ranks[pick].tolist())
        lo90 = mean - 1.6449 * sd_post
        hi90 = mean + 1.6449 * sd_post
        out_cov90.extend(((tt >= lo90) & (tt <= hi90)).tolist())
        lo50 = mean - 0.6745 * sd_post
        hi50 = mean + 0.6745 * sd_post
        out_cov50.extend(((tt >= lo50) & (tt <= hi50)).tolist())
        bhat_err.extend((b_e - b_true).tolist())
        Bhat_err.extend((B_e - B_true).tolist())
        if (rep + 1) % 40 == 0:
            print(f"  [{name}] rep {rep+1}/{R}")
    ranks = np.array(out_ranks)
    ks = stats.kstest(ranks, "uniform").pvalue
    res = {
        "scenario": name, "replicates": R, "EM": use_em,
        "coverage90": float(np.mean(out_cov90)),
        "coverage50": float(np.mean(out_cov50)),
        "rank_mean": float(ranks.mean()), "rank_sd": float(ranks.std()),
        "rank_sd_unif": float(np.sqrt(1/12)),
        "KS_p": float(ks),
        "b_mae": float(np.mean(np.abs(bhat_err))),
        "B_mae": float(np.mean(np.abs(Bhat_err))),
    }
    print(res)
    return res, ranks


def main():
    rows = []
    # EU-like: rich overlap
    res, ranks = run_scenario("EU-like", C=8, T=7, cover=[0.95, 0.8, 0.6],
                              R=120, use_em=True, seed=11)
    rows.append(res)
    np.save(RES / "s2_ranks_eu.npy", ranks)
    # CN-sparse: scarce observations
    res, ranks = run_scenario("CN-sparse", C=8, T=7, cover=[0.5, 0.35, 0.2],
                              R=120, use_em=True, seed=12)
    rows.append(res)
    np.save(RES / "s2_ranks_cn.npy", ranks)
    # oracle check (params known): should be perfectly calibrated
    res, ranks = run_scenario("EU-oracle", C=8, T=7, cover=[0.95, 0.8, 0.6],
                              R=120, use_em=False, seed=13)
    rows.append(res)
    np.save(RES / "s2_ranks_oracle.npy", ranks)
    pd.DataFrame(rows).to_csv(RES / "s2_sbc_summary.csv", index=False)
    print("\nsaved results/s2_sbc_summary.csv")


if __name__ == "__main__":
    main()
