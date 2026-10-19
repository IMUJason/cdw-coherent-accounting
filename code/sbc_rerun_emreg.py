"""Rerun the CHEAP SBC arms (oracle / EM-unregularized / EM-regularized / MAP)
on the SAME 32 datasets as sbc_prod_main (identical seeds regenerate identical
data), then merge with the stored full-Bayes column.

New arm:
  emreg : EM with a prior-scale ridge on (B, b) -- ridge toward (1, 0) with
          precision 1/0.25^2 and 1/0.4^2 (the model's own prior scale) -- and
          the anchor sigma now estimated (it is a free parameter in the
          production model). Tests a *properly regularized* plug-in that only
          misses hyperparameter integration.
The unregularized arm (already stored as em_*) is retained as the honest
"MLE-style plug-in is degenerate in this near-collinear design" data point.

Output: results/review_r5/sbc_prod_main.csv updated with emreg_* columns,
        summary json refreshed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sbc_production as sp  # noqa: E402
import w3_china_national_v2c as v2c  # noqa: E402

RES = ROOT / "results" / "review_r5"
R = 32
M = sp.M


def arm_emreg(phi_true, y, rng, max_iter=200, tol=1e-7):
    logB = np.zeros(M)
    b = np.zeros(M)
    log_sd = np.concatenate([[np.log(0.05)], [np.log(0.07)],
                             np.repeat(np.log(0.15), M - 2)])
    pm, P = v2c.theta_prior(phi_true["mu0"], phi_true["c"],
                            phi_true["rho"], phi_true["sd_eta"], sp.T)
    rB, rb = 1.0 / 0.25**2, 1.0 / 0.4**2     # prior-scale ridge -> (1, 0)
    it, delta = 0, np.inf
    while it < max_iter and delta > tol:
        m, cv = sp.cond_theta(pm, P, y, np.exp(logB), b, np.exp(log_sd))
        newB, newb, newsd = logB.copy(), b.copy(), log_sd.copy()
        # anchor sigma is free in the model -> estimate it (B=1, b=0 fixed)
        sel0 = np.where(sp.M_ARR == 0)[0]
        k0 = sp.T_ARR[sel0]
        resid0 = y[sel0] - m[k0]
        s2_0 = np.mean(resid0**2 + cv[k0, k0])
        newsd[0] = 0.5 * np.log(max(s2_0, 1e-8))
        for mi in range(1, M):
            sel = np.where(sp.M_ARR == mi)[0]
            k = sp.T_ARR[sel]
            Eth = m[k]
            Eth2 = Eth**2 + cv[k, k]
            yy = y[sel]
            A = np.array([[Eth2.sum() + rB, Eth.sum()],
                          [Eth.sum(), len(sel) + rb]])
            rhs = np.array([(yy * Eth).sum() + rB, yy.sum()])
            sol = np.linalg.solve(A, rhs)
            newB[mi], newb[mi] = sol
            if mi == 1:
                continue
            resid = yy - (sol[0] * Eth + sol[1])
            s2 = np.mean(resid**2 + sol[0]**2 * cv[k, k])
            newsd[mi] = 0.5 * np.log(max(s2, 1e-8))
        delta = max(np.abs(newB - logB).max(), np.abs(newb - b).max(),
                    np.abs(newsd - log_sd).max())
        logB, b, log_sd = newB, newb, newsd
        it += 1
    m, cv = sp.cond_theta(pm, P, y, np.exp(logB), b, np.exp(log_sd))
    d = np.stack([rng.multivariate_normal(m, cv) for _ in range(sp.N_CONDMIN)], 1)
    return d, it


df = pd.read_csv(RES / "sbc_prod_main.csv")
assert len(df) == R
emreg_cov, emreg_rank, emreg_it = [], [], []
for rep in range(R):
    rng = np.random.default_rng(91000 + rep)
    phi = sp.draw_phi(rng)
    theta = sp.sim_theta(phi, rng)
    y = sp.sim_y(phi, theta, rng)
    test_idx = rng.choice(sp.T, size=8, replace=False)
    d, it = arm_emreg(phi, y, rng)
    r, c = sp.metrics(d, theta, test_idx)
    emreg_cov.append(float(np.mean(c)))
    emreg_rank.append(float(np.mean(r)))
    emreg_it.append(it)
    print(f"rep {rep+1}/{R} emreg_cov={emreg_cov[-1]:.2f} iters={it}", flush=True)

df["emreg_cov"] = emreg_cov
df["emreg_rank_mean"] = emreg_rank
df["emreg_iters"] = emreg_it
df.to_csv(RES / "sbc_prod_main.csv", index=False)
summary = {}
for col in [c for c in df.columns if c.endswith("_cov")]:
    v = df[col].dropna()
    summary[col] = dict(mean=float(v.mean()),
                        mc_se=float(v.std(ddof=1) / np.sqrt(len(v))), n=int(len(v)))
for k in ["em_iters", "emreg_iters"]:
    if k in df.columns:
        summary[k + "_median"] = float(df[k].median())
summary["divergences_total"] = int(df["divergences"].sum())
(RES / "sbc_prod_main_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
print(json.dumps(summary, indent=2))
