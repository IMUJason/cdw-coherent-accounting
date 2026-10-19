"""R5-04/R5-06: matched SBC on the PRODUCTION China model.

All arms share the same simulated dataset per replicate.
  generator : phi ~ production priors (v2c), theta ~ AR(1)-increment DLM prior,
              y at the ACTUAL (m, t) cells of the production database.
  arms (main):
    oracle : exact conditional theta | phi_true
    em     : EM point estimates of measurement params (weak ridge 1e-3 for the
             n=1 sources), dynamics at TRUE values, then exact conditional
    map    : joint MAP of all hyperparameters (L-BFGS, 2 starts)
    full   : collapsed fully Bayesian NUTS (production make_model, y replaced,
             400+400)
  misspec : Gao errors generated AR(1) rho=0.7 (marginal sd 0.07), fitted with
            the standard independent-error model (full arm only).
  metrics : 8 uniformly sampled test states per replicate; per-replicate 90%
            coverage and mean rank; MC SE uses the replicate as unit (R5-05).

Usage: python sbc_production.py main 32 | misspec 16
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_china_national_v2c as v2c  # noqa: E402

RES = ROOT / "results" / "review_r5"
T = v2c.T
M = len(v2c.FIT_SOURCES)
SD0 = 0.6
N_CONDMIN = 400
OBS_DF = v2c.load_obs()
M_ARR = OBS_DF["m"].to_numpy().astype(int)
T_ARR = OBS_DF["t"].to_numpy().astype(int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw_phi(rng):
    return dict(
        mu0=rng.normal(np.log(26279.0), SD0),
        c=rng.normal(0.02, 0.03),
        rho=0.28 + 0.7 * sigmoid(rng.normal(0.0, 1.0)),
        sd_eta=abs(rng.normal(0.0, 0.12)),
        logB=np.concatenate([[0.0], rng.normal(0.0, 0.25, M - 1)]),
        b=np.concatenate([[0.0], rng.normal(0.0, 0.4, M - 1)]),
        log_sd_m=np.concatenate([[np.log(0.05)], [np.log(0.07)],
                                 rng.normal(np.log(0.15), 0.4, M - 2)]),
    )


def sim_theta(phi, rng):
    theta = np.empty(T)
    theta[0] = rng.normal(phi["mu0"], SD0)
    # stationary AR(1) initial increment, matching theta_prior
    g = rng.normal(phi["c"], phi["sd_eta"] / np.sqrt(1.0 - phi["rho"] ** 2))
    theta[1] = theta[0] + g
    for t in range(2, T):
        g = phi["c"] + phi["rho"] * (g - phi["c"]) + rng.normal(0, phi["sd_eta"])
        theta[t] = theta[t - 1] + g
    return theta


def sim_y(phi, theta, rng, gao_rho=0.0):
    B = np.exp(phi["logB"])
    sd = np.exp(phi["log_sd_m"])
    eps = rng.normal(0.0, sd[M_ARR])
    if gao_rho > 0:
        idx = np.where(M_ARR == 1)[0]
        order = np.argsort(T_ARR[idx])
        s_innov = sd[1] * np.sqrt(1.0 - gao_rho**2)
        e = np.empty(len(idx))
        e[order[0]] = rng.normal(0, sd[1])
        for k in range(1, len(order)):
            e[order[k]] = gao_rho * e[order[k - 1]] + rng.normal(0, s_innov)
        eps[idx[order]] = e
    return B[M_ARR] * theta[T_ARR] + phi["b"][M_ARR] + eps


def cond_theta(pm, P, y, B, b, sd_m):
    Pp = P.copy()
    r = P @ pm
    for i in range(len(y)):
        k = T_ARR[i]
        w = B[M_ARR[i]] / sd_m[M_ARR[i]] ** 2
        Pp[k, k] += B[M_ARR[i]] * w
        r[k] += (y[i] - b[M_ARR[i]]) * w
    cov = np.linalg.inv(Pp)
    return cov @ r, cov


def sample_cond(m, cv, rng):
    return np.stack([rng.multivariate_normal(m, cv) for _ in range(N_CONDMIN)], 1)


def arm_oracle(phi, y, rng):
    pm, P = v2c.theta_prior(phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"], T)
    m, cv = cond_theta(pm, P, y, np.exp(phi["logB"]), phi["b"],
                       np.exp(phi["log_sd_m"]))
    return sample_cond(m, cv, rng), None


def arm_em(phi_true, y, rng, max_iter=200, tol=1e-7):
    logB = np.zeros(M)
    b = np.zeros(M)
    log_sd = np.concatenate([[np.log(0.05)], [np.log(0.07)],
                             np.repeat(np.log(0.15), M - 2)])
    pm, P = v2c.theta_prior(phi_true["mu0"], phi_true["c"],
                            phi_true["rho"], phi_true["sd_eta"], T)
    it, delta = 0, np.inf
    while it < max_iter and delta > tol:
        m, cv = cond_theta(pm, P, y, np.exp(logB), b, np.exp(log_sd))
        newB, newb, newsd = logB.copy(), b.copy(), log_sd.copy()
        for mi in range(1, M):
            sel = np.where(M_ARR == mi)[0]
            if len(sel) == 0:
                continue
            k = T_ARR[sel]
            Eth = m[k]
            Eth2 = Eth**2 + cv[k, k]
            yy = y[sel]
            rB, rb = 1e-3, 1e-3          # weak ridge; n=1 sources stay near 0
            A = np.array([[Eth2.sum() + rB, Eth.sum()],
                          [Eth.sum(), len(sel) + rb]])
            rhs = np.array([(yy * Eth).sum(), yy.sum()])
            sol = np.linalg.solve(A, rhs)
            newB[mi], newb[mi] = sol
            if mi == 1:
                continue                   # gao sigma fixed at 0.07
            resid = yy - (sol[0] * Eth + sol[1])
            s2 = np.mean(resid**2 + sol[0]**2 * cv[k, k])
            newsd[mi] = 0.5 * np.log(max(s2, 1e-8))
        delta = max(np.abs(newB - logB).max(), np.abs(newb - b).max(),
                    np.abs(newsd - log_sd).max())
        logB, b, log_sd = newB, newb, newsd
        it += 1
    m, cv = cond_theta(pm, P, y, np.exp(logB), b, np.exp(log_sd))
    return sample_cond(m, cv, rng), it


def make_neg_logpost(y):
    mm = jnp.asarray(M_ARR)
    tt = jnp.asarray(T_ARR)
    yy = jnp.asarray(y)
    lsd_loc = jnp.log(jnp.concatenate([jnp.array([0.05]),
                                       jnp.repeat(0.15, M - 2)]))

    def nlpost(x):
        mu0, c, rho_raw, log_sd_eta = x[0], x[1], x[2], x[3]
        logB_f, b_f, lsd = x[4:10], x[10:16], x[16:22]
        sd_eta = jnp.exp(log_sd_eta)
        rho = 0.28 + 0.7 * jax.nn.sigmoid(rho_raw)
        # jax mirror of the AR(1)-increment prior precision (as in make_model)
        Dz = jnp.eye(T).at[jnp.arange(1, T), jnp.arange(0, T - 1)].set(-1.0)
        q = 1.0 / sd_eta**2
        Qz = jnp.zeros((T, T)).at[0, 0].set(1.0 / SD0**2)
        gidx = jnp.arange(1, T)
        Qz = Qz.at[gidx, gidx].set(q * (1 + rho**2))
        Qz = Qz.at[1, 1].set(q)
        Qz = Qz.at[T - 1, T - 1].set(q)
        offd = jnp.arange(2, T)
        Qz = Qz.at[offd, offd - 1].set(-q * rho)
        Qz = Qz.at[offd - 1, offd].set(-q * rho)
        P_np = Dz.T @ Qz @ Dz
        mz = jnp.concatenate([jnp.array([mu0]), jnp.full(T - 1, c)])
        pm = jnp.linalg.solve(Dz, mz)
        cov = jnp.linalg.inv(P_np)
        Bf = jnp.concatenate([jnp.zeros(1), logB_f])
        bf = jnp.concatenate([jnp.zeros(1), b_f])
        Bobs = jnp.exp(Bf)[mm]
        H = jnp.zeros((len(yy), T)).at[jnp.arange(len(yy)), tt].set(Bobs)
        mean_y = H @ pm + bf[mm]
        sd_all = jnp.concatenate([jnp.exp(lsd)[:1], jnp.array([0.07]),
                                  jnp.exp(lsd)[1:]])
        cov_y = H @ cov @ H.T + jnp.diag(sd_all[mm]**2)
        ll = dist.MultivariateNormal(mean_y, covariance_matrix=cov_y).log_prob(yy)
        lp = (dist.Normal(jnp.log(26279.0), SD0).log_prob(mu0)
              + dist.Normal(0.02, 0.03).log_prob(c)
              + dist.Normal(0.0, 1.0).log_prob(rho_raw)
              + dist.HalfNormal(0.12).log_prob(sd_eta)
              + dist.Normal(0.0, 0.25).log_prob(logB_f).sum()
              + dist.Normal(0.0, 0.4).log_prob(b_f).sum()
              + dist.Normal(lsd_loc, 0.4).log_prob(lsd).sum())
        return -(ll + lp)

    return nlpost


def arm_map(y, rng):
    nl = make_neg_logpost(y)
    grad = jax.jit(jax.grad(nl))
    x0 = np.concatenate([np.array([np.log(26279.0) + rng.normal(0, 0.1),
                                   0.02, 0.5, np.log(0.08)]),
                         np.zeros(6), np.zeros(6),
                         np.log(np.concatenate([[0.05], np.repeat(0.15, 5)]))])
    best, fbest = None, np.inf
    for s in range(2):
        xs = x0 + rng.normal(0, 0.05, x0.shape) * s
        res = minimize(lambda z: float(nl(jnp.asarray(z))), xs,
                       jac=lambda z: np.asarray(grad(jnp.asarray(z))),
                       method="L-BFGS-B", options=dict(maxiter=300))
        if res.fun < fbest:
            fbest, best = res.fun, res.x
    mu0, c, rho_raw, log_sd_eta = best[:4]
    rho = 0.28 + 0.7 * sigmoid(rho_raw)
    pm, P = v2c.theta_prior(mu0, c, rho, np.exp(log_sd_eta), T)
    B = np.exp(np.concatenate([[0.0], best[4:10]]))
    b = np.concatenate([[0.0], best[10:16]])
    sd = np.exp(np.concatenate([best[16:17], [np.log(0.07)], best[17:22]]))
    m, cv = cond_theta(pm, P, y, B, b, sd)
    return sample_cond(m, cv, rng)


def arm_full(y, seed):
    df = OBS_DF.copy()
    df["y"] = y
    mcmc = MCMC(NUTS(v2c.make_model(df, T), dense_mass=False),
                num_warmup=400, num_samples=400, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    smp = mcmc.get_samples()
    S = smp["mu0"].shape[0]
    draws = np.empty((T, S))
    rng = np.random.default_rng(seed)
    for s in range(S):
        phi = dict(mu0=float(smp["mu0"][s]), c=float(smp["c"][s]),
                   rho=float(smp["rho"][s]), sd_eta=float(smp["sd_eta"][s]),
                   logB=np.concatenate([[0.0], np.asarray(smp["logB_f"][s])]),
                   b=np.concatenate([[0.0], np.asarray(smp["b_f"][s])]),
                   log_sd_m=np.concatenate(
                       [[np.asarray(smp["log_sd_m_free"][s])[0]], [np.log(0.07)],
                        np.asarray(smp["log_sd_m_free"][s])[1:]]))
        m, cv = v2c.theta_posterior(phi, df, T)
        draws[:, s] = rng.multivariate_normal(m, cv)
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return draws, div


def metrics(draws, theta_true, test_idx):
    ranks, cov = [], []
    for k in test_idx:
        d = draws[k, :]
        ranks.append(float((d < theta_true[k]).mean()))
        lo, hi = np.quantile(d, [0.05, 0.95])
        cov.append(bool(lo <= theta_true[k] <= hi))
    return ranks, cov


def run(tag, R):
    rows = []
    t0 = time.time()
    for rep in range(R):
        rng = np.random.default_rng(91000 + rep)
        phi = draw_phi(rng)
        theta = sim_theta(phi, rng)
        y = sim_y(phi, theta, rng, gao_rho=0.7 if tag == "misspec" else 0.0)
        test_idx = rng.choice(T, size=8, replace=False)
        rec = dict(rep=rep)
        if tag == "main":
            d, _ = arm_oracle(phi, y, rng)
            r, c = metrics(d, theta, test_idx)
            rec["oracle_cov"] = float(np.mean(c))
            rec["oracle_rank_mean"] = float(np.mean(r))
            d, it = arm_em(phi, y, rng)
            r, c = metrics(d, theta, test_idx)
            rec["em_cov"] = float(np.mean(c))
            rec["em_rank_mean"] = float(np.mean(r))
            rec["em_iters"] = int(it)
            d = arm_map(y, rng)
            r, c = metrics(d, theta, test_idx)
            rec["map_cov"] = float(np.mean(c))
            rec["map_rank_mean"] = float(np.mean(r))
        d, div = arm_full(y, 42000 + rep)
        r, c = metrics(d, theta, test_idx)
        rec["full_cov"] = float(np.mean(c))
        rec["full_rank_mean"] = float(np.mean(r))
        rec["divergences"] = div
        rows.append(rec)
        print(f"[{tag}] rep {rep+1}/{R} "
              + " ".join(f"{k}={rec[k]:.2f}" for k in rec if k.endswith("_cov"))
              + f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"sbc_prod_{tag}.csv", index=False)
    summary = {}
    for col in [c for c in df.columns if c.endswith("_cov")]:
        v = df[col].dropna()
        summary[col] = dict(mean=float(v.mean()),
                            mc_se=float(v.std(ddof=1) / np.sqrt(len(v))),
                            n=int(len(v)))
    if "em_iters" in df.columns:
        summary["em_iters_median"] = float(df["em_iters"].median())
    summary["divergences_total"] = int(df["divergences"].sum())
    (RES / f"sbc_prod_{tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    run(tag, R)
