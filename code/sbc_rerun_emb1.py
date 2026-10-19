"""B=1 plug-in EM arm (R5-14-style proportional-bias plugin), rerun on the
same 32 datasets.

Motivation: with the latent level ~exp(10) and modest dynamics, the (B, b)
decomposition is numerically unidentified in EM-style joint fitting
(det of the 2x2 normal equations ~ 0 because sum E[theta^2] ~ (sum E theta)^2);
both unregularized and prior-ridge EM collapse (documented in
sbc_prod_main.csv em_* / emreg_*). The defensible plug-in fixes B = 1 and
estimates only a proportional offset b_m and noise sigma_m per source, with
dynamics hyperparameters at their TRUE values (the most generous plugin).
This isolates exactly the paper's claim: does ignoring hyperparameter
uncertainty (plus the fitted offsets) under-cover?

Output: adds emB1_* columns to results/review_r5/sbc_prod_main.csv.
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


def arm_emb1(phi_true, y, rng, max_iter=200, tol=1e-8):
    b = np.zeros(M)
    log_sd = np.concatenate([[np.log(0.05)], [np.log(0.07)],
                             np.repeat(np.log(0.15), M - 2)])
    pm, P = v2c.theta_prior(phi_true["mu0"], phi_true["c"],
                            phi_true["rho"], phi_true["sd_eta"], sp.T)
    rb = 1.0 / 0.4**2                     # prior-scale ridge on b -> 0
    it, delta = 0, np.inf
    while it < max_iter and delta > tol:
        m, cv = sp.cond_theta(pm, P, y, np.ones(M), b, np.exp(log_sd))
        newb, newsd = b.copy(), log_sd.copy()
        # anchor: B=1, b=0 fixed; sigma free
        sel0 = np.where(sp.M_ARR == 0)[0]
        k0 = sp.T_ARR[sel0]
        resid0 = y[sel0] - m[k0]
        newsd[0] = 0.5 * np.log(max(np.mean(resid0**2 + cv[k0, k0]), 1e-8))
        for mi in range(1, M):
            sel = np.where(sp.M_ARR == mi)[0]
            k = sp.T_ARR[sel]
            Eth = m[k]
            yy = y[sel]
            newb[mi] = (np.sum(yy - Eth) + rb * 0.0) / (len(sel) + rb)
            if mi == 1:
                continue                    # gao sigma fixed at 0.07 (model)
            resid = yy - (Eth + newb[mi])
            s2 = np.mean(resid**2 + cv[k, k])
            newsd[mi] = 0.5 * np.log(max(s2, 1e-8))
        delta = max(np.abs(newb - b).max(), np.abs(newsd - log_sd).max())
        b, log_sd = newb, newsd
        it += 1
    m, cv = sp.cond_theta(pm, P, y, np.ones(M), b, np.exp(log_sd))
    d = np.stack([rng.multivariate_normal(m, cv) for _ in range(sp.N_CONDMIN)], 1)
    return d, it


df = pd.read_csv(RES / "sbc_prod_main.csv")
covs, ranks, its = [], [], []
for rep in range(R):
    rng = np.random.default_rng(91000 + rep)
    phi = sp.draw_phi(rng)
    theta = sp.sim_theta(phi, rng)
    y = sp.sim_y(phi, theta, rng)
    test_idx = rng.choice(sp.T, size=8, replace=False)
    d, it = arm_emb1(phi, y, rng)
    r, c = sp.metrics(d, theta, test_idx)
    covs.append(float(np.mean(c)))
    ranks.append(float(np.mean(r)))
    its.append(it)
    print(f"rep {rep+1}/{R} emB1_cov={covs[-1]:.2f} iters={it}", flush=True)

df["emB1_cov"] = covs
df["emB1_rank_mean"] = ranks
df["emB1_iters"] = its
df.to_csv(RES / "sbc_prod_main.csv", index=False)
summary = {}
for col in [c for c in df.columns if c.endswith("_cov")]:
    v = df[col].dropna()
    summary[col] = dict(mean=float(v.mean()),
                        mc_se=float(v.std(ddof=1) / np.sqrt(len(v))), n=int(len(v)))
for k in ["em_iters", "emreg_iters", "emB1_iters"]:
    if k in df.columns:
        summary[k + "_median"] = float(df[k].median())
summary["divergences_total"] = int(df["divergences"].sum())
(RES / "sbc_prod_main_summary.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
print(json.dumps(summary, indent=2))
