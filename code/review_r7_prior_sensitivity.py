"""Bounded China measurement-prior sensitivity check (R7).

Runs only three arms differing in measurement prior scales, sequentially;
leaves production code/results and the manuscript untouched. All artifacts are
written under results/review_r7/prior/. Existing artifacts are never replaced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

# Set before importing JAX; also assert the effective setting below.
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS
import pandas as pd

import w3_china_national_v2c as cn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_r7" / "prior"
REFERENCE_THETA = ROOT / "results" / "china_theta_draws_v2c.npy"
BASE_SEED = 20261020
WARMUP = 800
SAMPLES_PER_CHAIN = 1000
CHAINS = 2
TARGET_ACCEPT = 0.95
FORECAST_PATHS_PER_DRAW = 20
ARMS = (
    {"name": "base", "logB_scale": 0.25, "offset_scale": 0.4, "seed": BASE_SEED},
    {"name": "tight", "logB_scale": 0.125, "offset_scale": 0.2, "seed": BASE_SEED + 1},
    {"name": "wide", "logB_scale": 0.5, "offset_scale": 0.8, "seed": BASE_SEED + 2},
)


def quantile_summary(values: np.ndarray, axis: int) -> np.ndarray:
    """Return median, 5th, and 95th percentiles in the requested axis."""
    return np.quantile(values, [0.5, 0.05, 0.95], axis=axis)


def checked_write_preflight() -> None:
    """Refuse to overwrite any existing R7 prior artifact."""
    if OUT.exists() and any(OUT.iterdir()):
        existing = sorted(p.name for p in OUT.iterdir())
        raise FileExistsError(
            f"Refusing to overwrite nonempty output directory {OUT}; "
            f"existing entries: {existing}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_THETA.is_file():
        raise FileNotFoundError(f"Required production reference draws missing: {REFERENCE_THETA}")


def load_fixed_theta_ref() -> float:
    draws = np.load(REFERENCE_THETA, allow_pickle=False)
    if draws.ndim != 2 or draws.shape[0] != len(cn.YEARS) or draws.shape[1] < 2:
        raise ValueError(
            f"Unexpected production theta shape {draws.shape}; expected "
            f"({len(cn.YEARS)}, S) with S >= 2"
        )
    if not np.isfinite(draws).all():
        bad = np.argwhere(~np.isfinite(draws))[:10].tolist()
        raise FloatingPointError(f"Nonfinite production theta values at indices {bad}")
    theta_ref = float(np.mean(draws, dtype=np.float64))
    if not np.isfinite(theta_ref):
        raise FloatingPointError(f"Nonfinite fixed theta_ref computed from {REFERENCE_THETA}")
    return theta_ref


def diagnostic_table(chain_samples: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, values in numpyro_summary(chain_samples, group_by_chain=True).items():
        for idx in np.ndindex(np.asarray(values["mean"]).shape):
            label = name + ("[" + ",".join(map(str, idx)) + "]" if idx else "")
            row = {"parameter": label}
            for key, value in values.items():
                scalar = np.asarray(value)[idx]
                row[key] = float(scalar) if np.issubdtype(np.asarray(scalar).dtype, np.number) else scalar.item()
            rows.append(row)
    if not rows:
        raise RuntimeError("NumPyro returned an empty diagnostic summary")
    return pd.DataFrame(rows)


def fit_arm(arm: dict[str, float | int | str], df: pd.DataFrame, theta_ref: float) -> dict:
    started = time.monotonic()
    name = str(arm["name"])
    seed = int(arm["seed"])
    state_seed = seed + 10000
    forecast_seed = seed + 20000
    logb_scale = float(arm["logB_scale"])
    offset_scale = float(arm["offset_scale"])
    log_sd_prior_sd = float(arm.get("log_sd_prior_sd", 0.4))
    sd_center_multiplier = float(arm.get("sd_center_multiplier", 1.0))

    model = cn.make_model(
        df, cn.T,
        logB_scale=logb_scale,
        offset_scale=offset_scale,
        log_sd_prior_sd=log_sd_prior_sd,
        sd_center_multiplier=sd_center_multiplier,
    )
    mcmc = MCMC(
        NUTS(model, target_accept_prob=TARGET_ACCEPT),
        num_warmup=WARMUP,
        num_samples=SAMPLES_PER_CHAIN,
        num_chains=CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    print(
        f"START {name}: logB_scale={logb_scale}, offset_scale={offset_scale}, "
        f"chains={CHAINS} sequential, warmup={WARMUP}, samples/chain={SAMPLES_PER_CHAIN}, "
        f"seed={seed}",
        flush=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))

    chain_samples_raw = mcmc.get_samples(group_by_chain=True)
    chain_samples = {key: np.asarray(value) for key, value in chain_samples_raw.items()}
    diagnostics = diagnostic_table(chain_samples)
    diagnostics.to_csv(OUT / f"{name}_diagnostics.csv", index=False)

    divergences = np.asarray(
        mcmc.get_extra_fields(group_by_chain=True)["diverging"], dtype=bool
    )
    if divergences.shape != (CHAINS, SAMPLES_PER_CHAIN):
        raise ValueError(
            f"{name}: unexpected divergence array shape {divergences.shape}; "
            f"expected {(CHAINS, SAMPLES_PER_CHAIN)}"
        )
    flat_samples = {key: value.reshape((-1,) + value.shape[2:]) for key, value in chain_samples.items()}
    required = {"mu0", "c", "rho", "sd_eta", "logB_f", "b_f", "log_sd_m_free"}
    missing = required.difference(flat_samples)
    if missing:
        raise KeyError(f"{name}: expected posterior fields missing: {sorted(missing)}")
    if any(not np.isfinite(value).all() for value in flat_samples.values()):
        bad = [key for key, value in flat_samples.items() if not np.isfinite(value).all()]
        raise FloatingPointError(f"{name}: nonfinite MCMC posterior field(s): {bad}")

    n_draws = CHAINS * SAMPLES_PER_CHAIN
    if flat_samples["mu0"].shape[0] != n_draws:
        raise ValueError(f"{name}: posterior draw count is not {n_draws}")
    logb_all = np.concatenate(
        [np.zeros((n_draws, 1)), flat_samples["logB_f"]], axis=1
    )
    b_all = np.concatenate(
        [np.zeros((n_draws, 1)), flat_samples["b_f"]], axis=1
    )

    # Apply one shared production-derived theta reference to every arm and draw.
    log_ratio = (np.exp(logb_all) - 1.0) * theta_ref + b_all
    with np.errstate(over="raise", invalid="raise"):
        ratios = np.exp(log_ratio)
    if not np.isfinite(ratios).all():
        raise FloatingPointError(f"{name}: nonfinite source ratios; inspect logB_f and b_f draws")
    ratio_q = quantile_summary(ratios, axis=0)
    pd.DataFrame({
        "source": cn.FIT_SOURCES,
        "anchor": [source == cn.ANCHOR for source in cn.FIT_SOURCES],
        "ratio_median": ratio_q[0],
        "ratio_p05": ratio_q[1],
        "ratio_p95": ratio_q[2],
        "theta_ref": theta_ref,
    }).to_csv(OUT / f"{name}_source_ratios.csv", index=False)

    # Draw latent annual theta conditionally for each hyperparameter posterior draw.
    theta_log = np.empty((cn.T, n_draws), dtype=np.float64)
    state_rng = np.random.default_rng(state_seed)
    log_sd_free = flat_samples["log_sd_m_free"]
    for i in range(n_draws):
        phi = {
            "mu0": float(flat_samples["mu0"][i]),
            "c": float(flat_samples["c"][i]),
            "rho": float(flat_samples["rho"][i]),
            "sd_eta": float(flat_samples["sd_eta"][i]),
            "logB": logb_all[i],
            "b": b_all[i],
            "log_sd_m": np.concatenate(
                [log_sd_free[i, :1], [np.log(0.07)], log_sd_free[i, 1:]]
            ),
        }
        conditional_mean, conditional_cov = cn.theta_posterior(phi, df, cn.T)
        conditional_cov = (conditional_cov + conditional_cov.T) * 0.5
        if not np.isfinite(conditional_mean).all() or not np.isfinite(conditional_cov).all():
            raise FloatingPointError(
                f"{name}: nonfinite conditional state Gaussian at flattened draw {i}; "
                f"mu0={phi['mu0']}, c={phi['c']}, rho={phi['rho']}, sd_eta={phi['sd_eta']}"
            )
        try:
            theta_log[:, i] = state_rng.multivariate_normal(
                conditional_mean, conditional_cov, check_valid="raise", tol=1e-8
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            eig_min = float(np.linalg.eigvalsh(conditional_cov).min())
            raise RuntimeError(
                f"{name}: failed conditional theta draw {i}; minimum covariance eigenvalue "
                f"={eig_min:.6g}; original error: {exc}"
            ) from exc
        if not np.isfinite(theta_log[:, i]).all():
            raise FloatingPointError(f"{name}: nonfinite theta state at flattened draw {i}")

    # Every posterior draw receives 20 independent mean-reverting continuations.
    horizon = 6
    future_log = np.empty((horizon, n_draws * FORECAST_PATHS_PER_DRAW), dtype=np.float64)
    forecast_rng = np.random.default_rng(forecast_seed)
    for i in range(n_draws):
        c = float(flat_samples["c"][i])
        rho = float(flat_samples["rho"][i])
        sd_eta = float(flat_samples["sd_eta"][i])
        growth = np.full(FORECAST_PATHS_PER_DRAW, theta_log[-1, i] - theta_log[-2, i])
        theta = np.full(FORECAST_PATHS_PER_DRAW, theta_log[-1, i])
        for k in range(horizon):
            growth = c + rho * (growth - c) + forecast_rng.normal(0.0, sd_eta, FORECAST_PATHS_PER_DRAW)
            theta = theta + growth
            future_log[k, i * FORECAST_PATHS_PER_DRAW:(i + 1) * FORECAST_PATHS_PER_DRAW] = theta
    with np.errstate(over="raise", invalid="raise"):
        posterior_gt = np.exp(theta_log) / 1e5
        forecast_gt = np.exp(future_log) / 1e5
    if not np.isfinite(posterior_gt).all() or not np.isfinite(forecast_gt).all():
        raise FloatingPointError(f"{name}: nonfinite exponentiated posterior or forecast values")

    annual_q = quantile_summary(posterior_gt, axis=1)
    pd.DataFrame({
        "year": cn.YEARS,
        "median_Gt": annual_q[0],
        "p05_Gt": annual_q[1],
        "p95_Gt": annual_q[2],
    }).to_csv(OUT / f"{name}_posterior_Gt.csv", index=False)
    forecast_q = quantile_summary(forecast_gt, axis=1)
    forecast_years = np.arange(2025, 2031)
    pd.DataFrame({
        "year": forecast_years,
        "median_Gt": forecast_q[0],
        "p05_Gt": forecast_q[1],
        "p95_Gt": forecast_q[2],
    }).to_csv(OUT / f"{name}_forecast_Gt.csv", index=False)

    np.save(OUT / f"{name}_theta_log_draws.npy", theta_log)
    np.save(OUT / f"{name}_forecast_Gt_draws.npy", forecast_gt)
    np.savez_compressed(OUT / f"{name}_hyperparameter_draws.npz", **chain_samples)

    key_year_records = []
    for year in (2023, 2024, 2030):
        if year <= cn.YEARS[-1]:
            year_q = annual_q[:, cn.YEARS.index(year)]
        else:
            year_q = forecast_q[:, int(year - 2025)]
        key_year_records.append({
            "arm": name, "year": year,
            "median_Gt": float(year_q[0]), "p05_Gt": float(year_q[1]), "p95_Gt": float(year_q[2]),
        })
    pd.DataFrame(key_year_records).to_csv(OUT / f"{name}_key_years.csv", index=False)

    finite_rhat = diagnostics["r_hat"].to_numpy(dtype=float)
    finite_ess = diagnostics["n_eff"].to_numpy(dtype=float)
    finite_rhat = finite_rhat[np.isfinite(finite_rhat)]
    finite_ess = finite_ess[np.isfinite(finite_ess)]
    if not len(finite_rhat) or not len(finite_ess):
        raise RuntimeError(f"{name}: no finite R-hat or ESS values in NumPyro summary")
    report = {
        "arm": name,
        "prior_parameters": {
            "logB_scale": logb_scale,
            "offset_scale": offset_scale,
            "log_sd_prior_sd": log_sd_prior_sd,
            "sd_center_multiplier": sd_center_multiplier,
            "mu0_scale_unchanged": 0.6,
            "other_model_settings_unchanged": True,
        },
        "sampler": {
            "chains": CHAINS,
            "chain_method": "sequential",
            "warmup_per_chain": WARMUP,
            "retained_per_chain": SAMPLES_PER_CHAIN,
            "target_accept": TARGET_ACCEPT,
            "seed": seed,
            "float64_enabled": bool(jax.config.x64_enabled),
        },
        "derived_random_seeds": {
            "conditional_state_draws": state_seed,
            "forecast_paths": forecast_seed,
        },
        "reference": {
            "theta_ref": theta_ref,
            "theta_ref_definition": "float64 mean of every value in production results/china_theta_draws_v2c.npy",
            "theta_ref_file": "results/china_theta_draws_v2c.npy",
            "ratio_formula": "exp((exp(logB)-1)*theta_ref+b)",
        },
        "observations": int(len(df)),
        "forecast_paths_per_posterior_draw": FORECAST_PATHS_PER_DRAW,
        "forecast_path_count": int(forecast_gt.shape[1]),
        "divergences_total": int(divergences.sum()),
        "divergences_by_chain": [int(v) for v in divergences.sum(axis=1)],
        "max_r_hat": float(finite_rhat.max()),
        "min_ess": float(finite_ess.min()),
        "runtime_seconds": float(time.monotonic() - started),
        "key_years_Gt": {
            str(row["year"]): [row["median_Gt"], row["p05_Gt"], row["p95_Gt"]]
            for row in key_year_records
        },
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "numpy_version": np.__version__,
        "theta_draws_file_shape": list(theta_log.shape),
        "forecast_draws_file_shape": list(forecast_gt.shape),
    }
    (OUT / f"{name}_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"DONE {name}: max_rhat={report['max_r_hat']:.6f}, "
        f"min_ess={report['min_ess']:.1f}, divergences={report['divergences_total']}, "
        f"2030={report['key_years_Gt']['2030']}, "
        f"seconds={report['runtime_seconds']:.1f}",
        flush=True,
    )
    return report


def main() -> None:
    checked_write_preflight()
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX float64 is not enabled; refusing to start sensitivity runs")
    if len(ARMS) != 3:
        raise RuntimeError(f"Expected exactly three specified prior arms, found {len(ARMS)}")

    df = cn.load_obs()
    if len(df) == 0 or not np.isfinite(df["y"].to_numpy(dtype=np.float64)).all():
        raise ValueError("China observation data is empty or contains nonfinite log values")
    theta_ref = load_fixed_theta_ref()
    print(
        f"CONFIG observations={len(df)}, theta_ref={theta_ref:.12f}, "
        f"JAX={jax.__version__}, NumPyro={numpyro.__version__}, float64={jax.config.x64_enabled}",
        flush=True,
    )

    reports = []
    key_year_tables = []
    for arm in ARMS:
        reports.append(fit_arm(arm, df, theta_ref))
        key_year_tables.append(pd.read_csv(OUT / f"{arm['name']}_key_years.csv"))
    combined = pd.concat(key_year_tables, ignore_index=True)
    combined.to_csv(OUT / "comparison_key_years.csv", index=False)
    (OUT / "comparison_run.json").write_text(
        json.dumps({
            "task": "bounded China measurement-prior sensitivity",
            "arms": reports,
            "prior_wide_distinction": (
                "The earlier review_r5 prior_wide arm changed only mu0_scale from 0.6 to 1.8; "
                "it left the measurement priors at logB_scale=0.25 and offset_scale=0.4. "
                "It was not a measurement-prior sensitivity arm."
            ),
            "interpretation_limit": (
                "A wide-prior arm estimates a posterior under that specified wide prior; "
                "it does not by itself prove source ratios are insensitive to prior choice."
            ),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("ALL ARMS COMPLETE", flush=True)
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
