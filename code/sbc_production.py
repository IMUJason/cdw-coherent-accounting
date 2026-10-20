"""Matched production-prior SBC with audited plug-in comparators.

All arms use the same generated data and eight test years per replicate.
EM arms know the true dynamic hyperparameters, estimate positive linear slopes,
offsets and free noise scales, and use explicitly stated linear-scale ridges.
The joint MAP is a mode in NumPyro's unconstrained sampling coordinates.
Results are isolated from legacy runs under results/review_r7/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.diagnostics import summary as chain_summary
from numpyro.infer import MCMC, NUTS
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_china_national_v2c as v2c

RES = ROOT / "results" / "review_r7"
T, M, SD0 = v2c.T, len(v2c.FIT_SOURCES), v2c.SD0
N_CONDMIN = 800
OBS_DF = v2c.load_obs()
M_ARR = OBS_DF["m"].to_numpy(dtype=int)
T_ARR = OBS_DF["t"].to_numpy(dtype=int)


def expand_noise(free):
    free = np.asarray(free)
    out = np.concatenate([free[..., :1], np.full(free.shape[:-1] + (1,), np.log(.07)),
                          free[..., 1:]], axis=-1)
    assert out.shape[-1] == M
    return out


def draw_phi(rng):
    noise_free = rng.normal(np.log(np.r_[.05, np.repeat(.15, M - 2)]), .4)
    return dict(mu0=rng.normal(np.log(26279.), .6), c=rng.normal(.02, .03),
                rho=.28 + .7 * expit(rng.normal()), sd_eta=abs(rng.normal(0, .12)),
                logB=np.r_[0., rng.normal(0, .25, M - 1)],
                b=np.r_[0., rng.normal(0, .4, M - 1)],
                log_sd_m=expand_noise(noise_free))


def sim_theta(phi, rng):
    pm, P = v2c.theta_prior(phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"], T)
    return rng.multivariate_normal(pm, np.linalg.inv(P))


def sim_y(phi, theta, rng, gao_rho=0.):
    sd = np.exp(phi["log_sd_m"])
    eps = rng.normal(0., sd[M_ARR])
    idx = np.flatnonzero(M_ARR == 1)
    idx = idx[np.argsort(T_ARR[idx])]
    if gao_rho and len(idx):
        e = rng.normal(0, sd[1])
        eps[idx[0]] = e
        for a, b in zip(idx[:-1], idx[1:]):
            decay = gao_rho ** int(T_ARR[b] - T_ARR[a])
            e = decay * e + rng.normal(0, sd[1] * np.sqrt(1 - decay**2))
            eps[b] = e
    return np.exp(phi["logB"])[M_ARR] * theta[T_ARR] + phi["b"][M_ARR] + eps


def cond_theta(pm, P, y, B, b, sd):
    H = np.zeros((len(y), T))
    H[np.arange(len(y)), T_ARR] = B[M_ARR]
    w = 1 / sd[M_ARR]**2
    precision = P + H.T @ (w[:, None] * H)
    cov = np.linalg.inv(precision)
    return cov @ (P @ pm + H.T @ (w * (y - b[M_ARR]))), (cov + cov.T) / 2


def sample_cond(m, cv, rng):
    return rng.multivariate_normal(m, cv, size=N_CONDMIN).T


def arm_oracle(phi, y, rng):
    pm, P = v2c.theta_prior(phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"], T)
    return sample_cond(*cond_theta(pm, P, y, np.exp(phi["logB"]), phi["b"],
                                   np.exp(phi["log_sd_m"])), rng)


def arm_em(phi, y, rng, ridge=False, max_iter=1000, tol=1e-6):
    B, b = np.ones(M), np.zeros(M)
    sd = np.r_[.05, .07, np.repeat(.15, M - 2)]
    pm, P = v2c.theta_prior(phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"], T)
    # Ridge penalties apply to B around one and b around zero, on their linear scales.
    rbias = (16., 6.25) if ridge else (1e-3, 1e-3)
    noise_floor_hits = 0
    for it in range(1, max_iter + 1):
        m, cv = cond_theta(pm, P, y, B, b, sd)
        newB, newb, newsd = B.copy(), b.copy(), sd.copy()
        for mi in range(1, M):
            ix = np.flatnonzero(M_ARR == mi)
            k, yy = T_ARR[ix], y[ix]
            ex, ex2 = m[k], m[k]**2 + cv[k, k]
            w = 1 / sd[mi]**2
            A = np.array([[w * ex2.sum() + rbias[0], w * ex.sum()],
                          [w * ex.sum(), w * len(ix) + rbias[1]]])
            rhs = np.array([w * (yy * ex).sum() + rbias[0], w * yy.sum()])
            slope, offset = np.linalg.solve(A, rhs)
            if slope < 1e-6:
                slope = 1e-6
                offset = w * np.sum(yy - slope * ex) / (w * len(ix) + rbias[1])
            newB[mi], newb[mi] = slope, offset
        for mi in range(M):
            if mi == 1:
                continue
            ix = np.flatnonzero(M_ARR == mi)
            k = T_ARR[ix]
            residual = y[ix] - newB[mi] * m[k] - newb[mi]
            variance = np.mean(residual**2 + newB[mi]**2 * cv[k, k])
            noise_floor_hits += int(variance < 1e-8)
            newsd[mi] = np.sqrt(max(variance, 1e-8))
        delta = max(np.max(np.abs(np.log(newB) - np.log(B))),
                    np.max(np.abs(newb - b)), np.max(np.abs(np.log(newsd / sd))))
        B, b, sd = newB, newb, newsd
        if delta < tol:
            break
    d = sample_cond(*cond_theta(pm, P, y, B, b, sd), rng)
    return d, dict(iters=it, converged=bool(delta < tol), delta=float(delta),
                   noise_floor_hits=noise_floor_hits, min_slope=float(B.min()))


def make_neg_logpost(y):
    mm, tt, yy = jnp.asarray(M_ARR), jnp.asarray(T_ARR), jnp.asarray(y)
    loc = jnp.log(jnp.r_[.05, jnp.repeat(.15, M - 2)])

    def nlpost(x):
        mu0, c, rho_raw, log_sd_eta = x[:4]
        lb, bf, lsd = x[4:4+M-1], x[4+M-1:4+2*(M-1)], x[4+2*(M-1):]
        sd_eta, rho = jnp.exp(log_sd_eta), .28 + .7 * jax.nn.sigmoid(rho_raw)
        D = jnp.eye(T).at[jnp.arange(1, T), jnp.arange(T-1)].set(-1.)
        q = 1 / sd_eta**2
        Q = jnp.zeros((T, T)).at[0, 0].set(1 / SD0**2)
        gidx = jnp.arange(1, T)
        Q = Q.at[gidx, gidx].set(q * (1 + rho**2)).at[1, 1].set(q).at[T-1, T-1].set(q)
        off = jnp.arange(2, T)
        Q = Q.at[off, off-1].set(-q*rho).at[off-1, off].set(-q*rho)
        cov = jnp.linalg.inv(D.T @ Q @ D)
        pm = jnp.linalg.solve(D, jnp.r_[mu0, jnp.repeat(c, T-1)])
        B = jnp.exp(jnp.r_[0., lb])
        b = jnp.r_[0., bf]
        sd = jnp.exp(jnp.r_[lsd[:1], jnp.log(.07), lsd[1:]])
        H = jnp.zeros((len(y), T)).at[jnp.arange(len(y)), tt].set(B[mm])
        ll = dist.MultivariateNormal(H @ pm + b[mm], covariance_matrix=
                                    H @ cov @ H.T + jnp.diag(sd[mm]**2)).log_prob(yy)
        lp = (dist.Normal(jnp.log(26279.), .6).log_prob(mu0)
              + dist.Normal(.02, .03).log_prob(c)
              + dist.Normal(0, 1).log_prob(rho_raw)
              + dist.HalfNormal(.12).log_prob(sd_eta) + log_sd_eta
              + dist.Normal(0, .25).log_prob(lb).sum()
              + dist.Normal(0, .4).log_prob(bf).sum()
              + dist.Normal(loc, .4).log_prob(lsd).sum())
        return -(ll + lp)
    return jax.jit(nlpost)


def arm_map(y, rng):
    nl = make_neg_logpost(y)
    grad = jax.jit(jax.grad(nl))
    x0 = np.r_[np.log(26279.), .02, 0., np.log(.08), np.zeros(2*(M-1)),
               np.log(np.r_[.05, np.repeat(.15, M-2)])]
    fits = []
    for i in range(2):
        xs = x0 + rng.normal(0, .05, len(x0)) * i
        fit = minimize(lambda x: float(nl(jnp.asarray(x))), xs,
                       jac=lambda x: np.asarray(grad(jnp.asarray(x))),
                       method="L-BFGS-B", options=dict(maxiter=1500, ftol=1e-12, gtol=1e-6))
        if np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise RuntimeError("No finite MAP solution")
    best = min(fits, key=lambda f: f.fun)
    x = best.x
    pm, P = v2c.theta_prior(x[0], x[1], .28 + .7*expit(x[2]), np.exp(x[3]), T)
    B = np.exp(np.r_[0., x[4:4+M-1]])
    b = np.r_[0., x[4+M-1:4+2*(M-1)]]
    sd = np.exp(expand_noise(x[4+2*(M-1):]))
    info = dict(success=bool(best.success), iterations=int(best.nit),
                gradient_max=float(np.max(np.abs(best.jac))), objective=float(best.fun))
    return sample_cond(*cond_theta(pm, P, y, B, b, sd), rng), info


def arm_full(y, seed):
    df = OBS_DF.copy()
    df["y"] = y
    mc = MCMC(NUTS(v2c.make_model(df, T), target_accept_prob=.9),
              num_warmup=600, num_samples=800, num_chains=2,
              chain_method="sequential", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed))
    samples = mc.get_samples()
    n = len(samples["mu0"])
    draws = np.empty((T, n))
    rng = np.random.default_rng(seed + 10000)
    for s in range(n):
        phi = dict(mu0=float(samples["mu0"][s]), c=float(samples["c"][s]),
                   rho=float(samples["rho"][s]), sd_eta=float(samples["sd_eta"][s]),
                   logB=np.r_[0., np.asarray(samples["logB_f"][s])],
                   b=np.r_[0., np.asarray(samples["b_f"][s])],
                   log_sd_m=expand_noise(samples["log_sd_m_free"][s]))
        cm, ck = v2c.theta_posterior(phi, df, T)
        draws[:, s] = rng.multivariate_normal(cm, ck)
    dg = chain_summary(mc.get_samples(group_by_chain=True), group_by_chain=True)
    rhats = np.concatenate([np.atleast_1d(v["r_hat"]).ravel() for v in dg.values()])
    esses = np.concatenate([np.atleast_1d(v["n_eff"]).ravel() for v in dg.values()])
    return draws, dict(divergences=int(mc.get_extra_fields()["diverging"].sum()),
                       max_rhat=float(np.nanmax(rhats)), min_ess=float(np.nanmin(esses)))


def metrics(draws, truth, test_idx):
    values = draws[test_idx]
    ranks = np.mean(values < truth[test_idx, None], axis=1)
    lo, hi = np.quantile(values, [.05, .95], axis=1)
    covered = (lo <= truth[test_idx]) & (truth[test_idx] <= hi)
    return ranks, covered


def run(tag, reps):
    RES.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    started = time.monotonic()
    for rep in range(reps):
        rng = np.random.default_rng(92000 + rep)
        phi = draw_phi(rng)
        truth = sim_theta(phi, rng)
        y = sim_y(phi, truth, rng, .7 if tag == "misspec" else 0.)
        test_idx = rng.choice(T, 8, replace=False)
        rec = dict(rep=rep, seed=92000+rep,
                   data_sha256=hashlib.sha256(y.tobytes()).hexdigest())
        details[f"truth_{rep}"] = truth
        details[f"y_{rep}"] = y
        details[f"test_idx_{rep}"] = test_idx
        details[f"phi_{rep}"] = np.r_[phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"],
                                         phi["logB"], phi["b"], phi["log_sd_m"]]
        arms = {}
        if tag == "main":
            arms["oracle"] = arm_oracle(phi, y, np.random.default_rng(101000+rep))
            for name, ridge in [("em", False), ("emreg", True)]:
                d, info = arm_em(phi, y, np.random.default_rng(102000+rep), ridge=ridge)
                arms[name] = d
                rec.update({name+"_"+k: v for k, v in info.items()})
            d, info = arm_map(y, np.random.default_rng(103000+rep))
            arms["map"] = d
            rec.update({"map_"+k: v for k, v in info.items()})
        d, info = arm_full(y, 43000+rep)
        arms["full"] = d
        rec.update(info)
        for name, draws in arms.items():
            ranks, cov = metrics(draws, truth, test_idx)
            rec[name+"_cov"] = float(cov.mean())
            rec[name+"_rank_mean"] = float(ranks.mean())
            rec[name+"_rank_one"] = float(ranks[0])
            details[f"{name}_ranks_{rep}"] = ranks
            details[f"{name}_covered_{rep}"] = cov
        rows.append(rec)
        pd.DataFrame(rows).to_csv(RES / f"sbc_{tag}.csv", index=False)
        np.savez_compressed(RES / f"sbc_{tag}_details.npz", **details)
        print(f"{tag} {rep+1}/{reps} " + " ".join(f"{a}={rec[a+'_cov']:.3f}" for a in arms)
              + f" div={info['divergences']} elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
    df = pd.DataFrame(rows)
    report = dict(generator="production priors including random anchor noise; exact AR prior",
                  test_states_per_replicate=8, replicates=reps, warmup_per_chain=600,
                  retained_per_chain=800, chains=2, divergences=int(df.divergences.sum()),
                  max_rhat=float(df.max_rhat.max()), min_ess=float(df.min_ess.min()),
                  seconds=time.monotonic()-started)
    for name in arms:
        cov, ranks = df[name+"_cov"], df[name+"_rank_mean"]
        report[name] = dict(coverage=float(cov.mean()), mc_se=float(cov.std(ddof=1)/np.sqrt(reps)),
                            rank_mean=float(ranks.mean()),
                            rank_mc_se=float(ranks.std(ddof=1)/np.sqrt(reps)))
    report["em_converged"] = int(df.em_converged.sum()) if "em_converged" in df else None
    report["emreg_converged"] = int(df.emreg_converged.sum()) if "emreg_converged" in df else None
    report["map_successes"] = int(df.map_success.sum()) if "map_success" in df else None
    (RES / f"sbc_{tag}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", choices=["main", "misspec"])
    parser.add_argument("replicates", nargs="?", type=int, default=32)
    opts = parser.parse_args()
    if opts.replicates < 2:
        parser.error("At least two replicates are required to estimate Monte Carlo error")
    run(opts.tag, opts.replicates)
