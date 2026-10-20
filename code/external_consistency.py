"""Deterministic China anchor holdout and same-source consistency check.

The 2024 statutory anchor is excluded before fitting. For the held-out year,
report a latent-state credible interval and an anchor-report observation
predictive interval separately. The Gao comparison is explicitly a
same-source consistency check, not an independent validation.
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
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import w3_china_national_v2c as v2c  # noqa: E402

SEED = 51
STATE_SEED = 51001
OBS_SEED = 51002
WARMUP = 1500
SAMPLES = 1500


def main() -> None:
    started = time.monotonic()
    df = v2c.load_obs()
    if "source_id" not in df:
        raise KeyError("Expected source_id to identify the held-out 2024 anchor")
    mask = (df["source_id"] == "meescc_2025") & (df["year"] == 2024)
    held = df.loc[mask].copy()
    if len(held) != 1:
        raise ValueError(f"Expected exactly one 2024 anchor observation; found {len(held)}")
    truth_wt = float(held.iloc[0]["value_wt"])
    if not np.isclose(truth_wt, 231000.0):
        raise ValueError(f"Unexpected held-out 2024 anchor value: {truth_wt}")
    df_tr = df.loc[~mask].reset_index(drop=True)
    print(f"[A] withheld {len(held)} observation; fitting on {len(df_tr)}", flush=True)

    mcmc = MCMC(
        NUTS(v2c.make_model(df_tr, v2c.T), target_accept_prob=0.9),
        num_warmup=WARMUP,
        num_samples=SAMPLES,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(SEED), extra_fields=("diverging",))
    samples = {key: np.asarray(value) for key, value in mcmc.get_samples().items()}
    take = np.arange(0, SAMPLES, 2)
    logb_all = np.concatenate([np.zeros((len(take), 1)), samples["logB_f"][take]], axis=1)
    b_all = np.concatenate([np.zeros((len(take), 1)), samples["b_f"][take]], axis=1)
    log_sd_free = samples["log_sd_m_free"]
    theta_refit = np.empty((v2c.T, len(take)), dtype=np.float64)
    rng_state = np.random.default_rng(STATE_SEED)
    rng_obs = np.random.default_rng(OBS_SEED)
    anchor_observation = np.empty(len(take), dtype=np.float64)

    for j, s in enumerate(take):
        phi = {
            "mu0": float(samples["mu0"][s]),
            "c": float(samples["c"][s]),
            "rho": float(samples["rho"][s]),
            "sd_eta": float(samples["sd_eta"][s]),
            "logB": logb_all[j],
            "b": b_all[j],
            "log_sd_m": np.concatenate(
                [log_sd_free[s, :1], [np.log(0.07)], log_sd_free[s, 1:]]
            ),
        }
        mean, cov = v2c.theta_posterior(phi, df_tr, v2c.T)
        theta_refit[:, j] = rng_state.multivariate_normal(mean, cov)
        anchor_observation[j] = theta_refit[v2c.YEARS.index(2024), j] + rng_obs.normal(
            0.0, np.exp(phi["log_sd_m"][0])
        )

    i24 = v2c.YEARS.index(2024)
    latent_wt = np.exp(theta_refit[i24])
    observation_wt = np.exp(anchor_observation)
    truth_gt = truth_wt / 1e5
    latent_q = np.quantile(latent_wt / 1e5, [0.05, 0.5, 0.95])
    obs_q = np.quantile(observation_wt / 1e5, [0.05, 0.5, 0.95])
    divergences = np.asarray(mcmc.get_extra_fields()["diverging"], dtype=bool)
    diag = numpyro_summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
    rhats = np.concatenate([np.atleast_1d(v["r_hat"]).ravel() for v in diag.values()])
    esses = np.concatenate([np.atleast_1d(v["n_eff"]).ravel() for v in diag.values()])
    full = np.load(RES / "china_theta_draws_v2c.npy", allow_pickle=False)
    if full.shape[0] != v2c.T:
        raise ValueError(f"Unexpected production state draw shape: {full.shape}")
    i23 = v2c.YEARS.index(2023)
    gao_percentile = float((np.exp(full[i23]) < 239000.0).mean())
    gao_median_gt = float(np.median(np.exp(full[i23])) / 1e5)

    rows = [
        dict(test="A_holdout_2024_latent", target="anchor-referenced latent state",
             seed=SEED, state_seed=STATE_SEED, observation_seed=None,
             withheld_obs=1, truth_Gt=truth_gt, median_Gt=latent_q[1],
             lo90_Gt=latent_q[0], hi90_Gt=latent_q[2],
             covered=bool(latent_q[0] <= truth_gt <= latent_q[2])),
        dict(test="A_holdout_2024_observation", target="future anchor report",
             seed=SEED, state_seed=STATE_SEED, observation_seed=OBS_SEED,
             withheld_obs=1, truth_Gt=truth_gt, median_Gt=obs_q[1],
             lo90_Gt=obs_q[0], hi90_Gt=obs_q[2],
             covered=bool(obs_q[0] <= truth_gt <= obs_q[2])),
        dict(test="B_Gao2023_same_source", target="published forecast consistency",
             seed=None, state_seed=None, observation_seed=None,
             withheld_obs=0, truth_Gt=2.39, median_Gt=gao_median_gt,
             lo90_Gt=np.nan, hi90_Gt=np.nan, covered=np.nan),
    ]
    pd.DataFrame(rows).to_csv(RES / "external_consistency_r7.csv", index=False)
    metadata = {
        "fit": {"observations": int(len(df_tr)), "withheld_observations": 1,
                "withheld_source_id": "meescc_2025", "withheld_year": 2024,
                "withheld_value_Gt": truth_gt, "seed": SEED,
                "warmup": WARMUP, "retained_draws": SAMPLES, "chains": 1,
                "target_accept": 0.9, "jax_x64": bool(jax.config.x64_enabled)},
        "conditional_draws": {"retained_hyper_draws": int(len(take)),
                               "latent_state_seed": STATE_SEED,
                               "anchor_observation_seed": OBS_SEED},
        "diagnostics": {"divergences": int(divergences.sum()),
                        "max_r_hat": float(np.nanmax(rhats)),
                        "min_ess": float(np.nanmin(esses))},
        "holdout": {"latent_state": {"median_Gt": float(latent_q[1]),
                                       "lo90_Gt": float(latent_q[0]),
                                       "hi90_Gt": float(latent_q[2])},
                    "future_anchor_report": {"median_Gt": float(obs_q[1]),
                                              "lo90_Gt": float(obs_q[0]),
                                              "hi90_Gt": float(obs_q[2])}},
        "same_source_check": {"source": "Gao et al. (2025)", "year": 2023,
                              "published_forecast_Gt": 2.39,
                              "production_posterior_median_Gt": gao_median_gt,
                              "published_forecast_posterior_percentile": gao_percentile,
                              "interpretation": "consistency only; the historical Gao series is in the fit"},
        "runtime_seconds": float(time.monotonic() - started),
        "jax_version": jax.__version__, "numpyro_version": numpyro.__version__,
        "numpy_version": np.__version__,
        "output": "results/external_consistency_r7.csv",
    }
    (RES / "external_consistency_r7.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        RES / "external_consistency_r7_draws.npz",
        theta_latent_wt=theta_refit, anchor_observation_wt=observation_wt,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
