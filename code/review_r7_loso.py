"""Leave-one-source-out influence diagnostic for the China reconstruction.

Refits the production model with each measurement process removed in turn
and reports the 2023/2024 posterior and the 2030 forecast. Isolated outputs
under results/review_r7/loso/.
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
sys.path.insert(0, str(ROOT / "scripts"))
import w3_china_national_v2c as v2c  # noqa: E402

OUT = ROOT / "results" / "review_r7" / "loso"
BASE_SEED = 61000
WARMUP = 1000
SAMPLES = 1000
TARGET_ACCEPT = 0.9


def main() -> None:
    started = time.monotonic()
    df = v2c.load_obs()
    rows = []
    for i, src in enumerate(v2c.FIT_SOURCES):
        drop = df["src"] == src
        sub = df[~drop].reset_index(drop=True)
        seed = BASE_SEED + i
        mcmc = MCMC(NUTS(v2c.make_model(sub, v2c.T), target_accept_prob=TARGET_ACCEPT),
                    num_warmup=WARMUP, num_samples=SAMPLES, num_chains=1,
                    progress_bar=False)
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
            mean, cov = v2c.theta_posterior(phi, sub, v2c.T)
            theta[:, j] = rng.multivariate_normal(mean, cov)
        gt = np.exp(theta) / 1e5
        rec = dict(dropped=src, dropped_obs=int(drop.sum()), remaining_obs=len(sub),
                   seed=seed,
                   div=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
        for year in (2023, 2024):
            q = np.quantile(gt[v2c.YEARS.index(year)], [0.05, 0.5, 0.95])
            rec[f"y{year}_median"] = float(q[1])
            rec[f"y{year}_lo"] = float(q[0])
            rec[f"y{year}_hi"] = float(q[2])
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
        q = np.quantile(fgt[5], [0.05, 0.5, 0.95])
        rec["y2030_median"] = float(q[1])
        rec["y2030_lo"] = float(q[0])
        rec["y2030_hi"] = float(q[2])
        rows.append(rec)
        print(f"drop {src:16s} n={int(drop.sum())} "
              f"2023={rec['y2023_median']:.3f} [{rec['y2023_lo']:.2f},{rec['y2023_hi']:.2f}] "
              f"2030={rec['y2030_median']:.3f} div={rec['div']} "
              f"({time.monotonic()-started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "loso_summary.csv", index=False)
    meta = {
        "purpose": "leave-one-source-out influence diagnostic",
        "warmup": WARMUP, "retained_draws": SAMPLES, "chains": 1,
        "target_accept": TARGET_ACCEPT, "float64": bool(jax.config.x64_enabled),
        "total_divergences": int(frame["div"].sum()),
        "runtime_seconds": float(time.monotonic() - started),
        "jax_version": jax.__version__, "numpyro_version": numpyro.__version__,
    }
    (OUT / "loso_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
