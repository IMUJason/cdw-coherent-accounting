"""Dynamic-prior sensitivity for the China model (kappa and sigma_eta).

Three arms: production priors, doubled kappa prior sd (0.03 -> 0.06), and
doubled sigma_eta prior scale (0.12 -> 0.24). Everything else is unchanged.
Outputs isolated under results/review_r7/dynprior/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import w3_china_national_v2c as v2c  # noqa: E402

OUT = ROOT / "results" / "review_r7" / "dynprior"
ARMS = (
    {"arm": "base", "kappa_sd": 0.03, "sd_eta_scale": 0.12, "seed": 63001},
    {"arm": "kappa_wide", "kappa_sd": 0.06, "sd_eta_scale": 0.12, "seed": 63002},
    {"arm": "sigma_wide", "kappa_sd": 0.03, "sd_eta_scale": 0.24, "seed": 63003},
)
WARMUP = 1000
SAMPLES = 1000
TARGET_ACCEPT = 0.9


def main() -> None:
    started = time.monotonic()
    df = v2c.load_obs()
    rows = []
    for arm in ARMS:
        seed = int(arm["seed"])
        mcmc = MCMC(
            NUTS(v2c.make_model(df, v2c.T, kappa_sd=float(arm["kappa_sd"]),
                                sd_eta_scale=float(arm["sd_eta_scale"])),
                 target_accept_prob=TARGET_ACCEPT),
            num_warmup=WARMUP, num_samples=SAMPLES, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed))
        smp = mcmc.get_samples()
        take = np.arange(0, SAMPLES, 2)
        S = len(take)
        logB_all = np.concatenate([np.zeros((S, 1)),
                                   np.asarray(smp["logB_f"])[take]], axis=1)
        b_all = np.concatenate([np.zeros((S, 1)), np.asarray(smp["b_f"])[take]], axis=1)
        lsd = np.asarray(smp["log_sd_m_free"])
        theta = np.empty((v2c.T, S))
        rng = np.random.default_rng(seed + 1000)
        for j, s in enumerate(take):
            phi = dict(mu0=float(smp["mu0"][s]), c=float(smp["c"][s]),
                       rho=float(smp["rho"][s]), sd_eta=float(smp["sd_eta"][s]),
                       logB=logB_all[j], b=b_all[j],
                       log_sd_m=np.concatenate([lsd[s, :1], [np.log(0.07)], lsd[s, 1:]]))
            mean, cov = v2c.theta_posterior(phi, df, v2c.T)
            theta[:, j] = rng.multivariate_normal(mean, cov)
        gt = np.exp(theta) / 1e5
        fut = np.empty((6, S))
        for j in range(S):
            c_ = float(smp["c"][take[j]])
            rho_ = float(smp["rho"][take[j]])
            sd_ = float(smp["sd_eta"][take[j]])
            g = theta[-1, j] - theta[-2, j]
            th = theta[-1, j]
            for k in range(6):
                g = c_ + rho_ * (g - c_) + rng.normal(0, sd_)
                th = th + g
                fut[k, j] = th
        fgt = np.exp(fut) / 1e5
        rec = dict(arm=arm["arm"], kappa_sd=arm["kappa_sd"],
                   sd_eta_scale=arm["sd_eta_scale"], seed=seed,
                   kappa_post_median=float(np.median(np.asarray(smp["c"]))),
                   rho_post_median=float(np.median(np.asarray(smp["rho"]))),
                   sd_eta_post_median=float(np.median(np.asarray(smp["sd_eta"]))),
                   div=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
        for year in (2023, 2024):
            q = np.quantile(gt[v2c.YEARS.index(year)], [0.05, 0.5, 0.95])
            rec[f"y{year}_median"] = float(q[1])
            rec[f"y{year}_lo"] = float(q[0])
            rec[f"y{year}_hi"] = float(q[2])
        q = np.quantile(fgt[5], [0.05, 0.5, 0.95])
        rec["y2030_median"] = float(q[1])
        rec["y2030_lo"] = float(q[0])
        rec["y2030_hi"] = float(q[2])
        rows.append(rec)
        print(f"{arm['arm']:12s} 2023={rec['y2023_median']:.3f} "
              f"2030={rec['y2030_median']:.3f} [{rec['y2030_lo']:.2f},{rec['y2030_hi']:.2f}] "
              f"div={rec['div']} ({time.monotonic()-started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "dynprior_summary.csv", index=False)
    meta = {
        "purpose": "dynamic-prior sensitivity (kappa sd, sigma_eta scale)",
        "arms": [{k: arm[k] for k in ("arm", "kappa_sd", "sd_eta_scale", "seed")} for arm in ARMS],
        "warmup": WARMUP, "retained_draws": SAMPLES, "chains": 1,
        "target_accept": TARGET_ACCEPT, "float64": bool(jax.config.x64_enabled),
        "total_divergences": int(frame["div"].sum()),
        "runtime_seconds": float(time.monotonic() - started),
        "jax_version": jax.__version__, "numpyro_version": numpyro.__version__,
    }
    (OUT / "dynprior_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
