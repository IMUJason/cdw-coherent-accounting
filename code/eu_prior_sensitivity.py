"""Bounded two-arm EU prior-location sensitivity (R7), repro-package layout.

Fits the full 27-member, seven-wave panel under the production prior
location (-0.2) and the sensitivity location (0.0). No rolling backtests;
outputs only under results/review_r7/eu_prior/.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import eu_testbed as eu  # noqa: E402

OUT = ROOT / "results" / "review_r7" / "eu_prior"
ARMS = (
    {"arm": "baseline", "logB_loc": -0.2, "seed": 20261020},
    {"arm": "loc0", "logB_loc": 0.0, "seed": 20261021},
)
WARMUP = 600
SAMPLES_PER_CHAIN = 600
CHAINS = 2
TARGET_ACCEPT = 0.9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def diag_frame(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, values in numpyro_summary(samples, group_by_chain=True).items():
        for idx in np.ndindex(np.asarray(values["mean"]).shape):
            label = name + ("[" + ",".join(map(str, idx)) + "]" if idx else "")
            row = {"parameter": label}
            for key, value in values.items():
                scalar = np.asarray(value)[idx]
                row[key] = float(scalar) if np.issubdtype(np.asarray(scalar).dtype, np.number) else scalar.item()
            rows.append(row)
    if not rows:
        raise RuntimeError("NumPyro returned no parameter diagnostics")
    return pd.DataFrame(rows)


def posterior_state_draws(samples: dict[str, np.ndarray], df: pd.DataFrame,
                          C: int, T: int, seed: int) -> np.ndarray:
    n = samples["mu0"].shape[0]
    out = np.empty((C * T, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i in range(n):
        phi = {
            "mu0": np.asarray(samples["mu0"][i], dtype=np.float64),
            "drift": float(samples["drift"][i]),
            "sd_theta": float(np.exp(samples["log_sd_theta"][i])),
            "logB_F": float(samples["logB_F"][i]),
            "b_F": float(samples["b_F"][i]),
            "sd_m": np.exp(np.asarray(samples["log_sd_m"][i], dtype=np.float64)),
        }
        mean, cov = eu.theta_posterior(phi, df, C, T)
        cov = (cov + cov.T) * 0.5
        out[:, i] = rng.multivariate_normal(mean, cov, check_valid="raise", tol=1e-8)
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite latent posterior states")
    return out


def fit_arm(arm: dict, df: pd.DataFrame, level0: list[float], C: int, T: int) -> dict:
    name = arm["arm"]
    start = time.monotonic()
    model = eu.make_model(df, C, T, level0, logB_loc=float(arm["logB_loc"]))
    mcmc = MCMC(
        NUTS(model, target_accept_prob=TARGET_ACCEPT),
        num_warmup=WARMUP,
        num_samples=SAMPLES_PER_CHAIN,
        num_chains=CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    print(f"START {name}: prior logB_loc={arm['logB_loc']}, chains={CHAINS}, "
          f"warmup={WARMUP}, draws/chain={SAMPLES_PER_CHAIN}, seed={arm['seed']}", flush=True)
    mcmc.run(jax.random.PRNGKey(int(arm["seed"])), extra_fields=("diverging",))
    chains = {key: np.asarray(value, dtype=np.float64)
              for key, value in mcmc.get_samples(group_by_chain=True).items()}
    flat = {key: value.reshape((-1,) + value.shape[2:]) for key, value in chains.items()}
    if any(not np.isfinite(value).all() for value in flat.values()):
        raise FloatingPointError(f"Nonfinite posterior hyperparameter in {name}")

    diagnostics = diag_frame(chains)
    diagnostics.to_csv(OUT / f"{name}_diagnostics.csv", index=False)
    finite_rhat = diagnostics["r_hat"].to_numpy(dtype=np.float64)
    finite_ess = diagnostics["n_eff"].to_numpy(dtype=np.float64)
    finite_rhat = finite_rhat[np.isfinite(finite_rhat)]
    finite_ess = finite_ess[np.isfinite(finite_ess)]
    if not len(finite_rhat) or not len(finite_ess):
        raise RuntimeError(f"Missing finite R-hat or ESS for {name}")

    divergence = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"], dtype=bool)
    if divergence.shape != (CHAINS, SAMPLES_PER_CHAIN):
        raise ValueError(f"Unexpected divergence shape {divergence.shape} for {name}")
    state_seed = int(arm["seed"]) + 10000
    theta_log = posterior_state_draws(flat, df, C, T, state_seed)

    return {
        "arm": name,
        "logB_loc": float(arm["logB_loc"]),
        "seed": int(arm["seed"]),
        "state_draw_seed": state_seed,
        "samples": flat,
        "theta_log": theta_log,
        "diagnostics": {
            "divergences_total": int(divergence.sum()),
            "divergences_by_chain": [int(x) for x in divergence.sum(axis=1)],
            "max_r_hat": float(finite_rhat.max()),
            "min_ess": float(finite_ess.min()),
        },
        "runtime_seconds": float(time.monotonic() - start),
    }


def summarize_states(arms: list[dict], common_theta_ref: float, years: list[int],
                     empirical_ratio_median: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratio_rows, member_rows, aggregate_rows = [], [], []
    C, T = len(eu.EU27), len(years)
    for fit in arms:
        name = fit["arm"]
        samples = fit["samples"]
        theta = fit["theta_log"].reshape(C, T, -1)
        slope = np.exp(samples["logB_F"])
        ratio = np.exp((slope - 1.0) * common_theta_ref + samples["b_F"])
        rq = np.quantile(ratio, [0.5, 0.05, 0.95])
        ratio_rows.append({
            "arm": name,
            "logB_loc": fit["logB_loc"],
            "theta_ref_log_tonnes": common_theta_ref,
            "ratio_median": float(rq[0]),
            "ratio_p05": float(rq[1]),
            "ratio_p95": float(rq[2]),
            "slope_B_median": float(np.median(slope)),
            "slope_B_p05": float(np.quantile(slope, 0.05)),
            "slope_B_p95": float(np.quantile(slope, 0.95)),
            "offset_b_median": float(np.median(samples["b_F"])),
            "empirical_F_TOTAL_ratio_median": empirical_ratio_median,
        })
        member_Mt = np.exp(theta) / 1e6
        q_member = np.quantile(member_Mt, [0.5, 0.05, 0.95], axis=2)
        for ci, geo in enumerate(eu.EU27):
            for ti, year in enumerate(years):
                member_rows.append({
                    "arm": name, "geo": geo, "year": year,
                    "median_Mt": float(q_member[0, ci, ti]),
                    "p05_Mt": float(q_member[1, ci, ti]),
                    "p95_Mt": float(q_member[2, ci, ti]),
                })
        agg_Mt = member_Mt.sum(axis=0)
        q_agg = np.quantile(agg_Mt, [0.5, 0.05, 0.95], axis=1)
        for ti, year in enumerate(years):
            aggregate_rows.append({
                "arm": name, "year": year,
                "median_Mt": float(q_agg[0, ti]),
                "p05_Mt": float(q_agg[1, ti]),
                "p95_Mt": float(q_agg[2, ti]),
            })
    return pd.DataFrame(ratio_rows), pd.DataFrame(member_rows), pd.DataFrame(aggregate_rows)


def add_arm_comparison(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    base = df[df["arm"] == "baseline"].drop(columns="arm").set_index(keys)
    alt = df[df["arm"] == "loc0"].drop(columns="arm").set_index(keys)
    joined = base.add_suffix("_baseline").join(alt.add_suffix("_loc0"), how="inner")
    joined["median_pct_change"] = 100.0 * (joined["median_Mt_loc0"] / joined["median_Mt_baseline"] - 1.0)
    return joined.reset_index()


def main() -> None:
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX float64 is not enabled")
    output_names = {
        "baseline_diagnostics.csv", "loc0_diagnostics.csv", "source_ratio_at_common_theta.csv",
        "member_levels_Mt.csv", "aggregate_levels_Mt.csv", "member_level_comparison.csv",
        "aggregate_level_comparison.csv", "run_metadata.json",
    }
    existing = sorted(name for name in output_names if (OUT / name).exists())
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing EU sensitivity outputs: {existing}")

    df, years_raw = eu.load_panel()
    years = [int(y) for y in years_raw]
    C, T = len(eu.EU27), len(years)
    first_year = years[0]
    first = df[(df["m"] == 0) & (df["time"] == first_year)].set_index("c")["y"]
    fallback = float(df[df["m"] == 0]["y"].median())
    level0 = [float(first.get(c, fallback)) for c in range(C)]
    if len(df) != 376 or (C, T) != (27, 7):
        raise ValueError(f"Panel dimensions changed: obs={len(df)}, C={C}, T={T}")

    OUT.mkdir(parents=True, exist_ok=True)
    runs = [fit_arm(arm, df, level0, C, T) for arm in ARMS]
    common_theta_ref = float(np.median(runs[0]["theta_log"]))
    tot = df[df["m"] == 0][["c", "t", "y"]].rename(columns={"y": "y_tot"})
    construction = df[df["m"] == 1][["c", "t", "y"]].rename(columns={"y": "y_f"})
    paired = tot.merge(construction, on=["c", "t"], how="inner")
    empirical_ratio_median = float(np.median(np.exp(paired["y_f"] - paired["y_tot"])))
    ratios, members, aggregates = summarize_states(runs, common_theta_ref, years, empirical_ratio_median)
    ratios.to_csv(OUT / "source_ratio_at_common_theta.csv", index=False)
    members.to_csv(OUT / "member_levels_Mt.csv", index=False)
    aggregates.to_csv(OUT / "aggregate_levels_Mt.csv", index=False)
    add_arm_comparison(members, ["geo", "year"]).to_csv(OUT / "member_level_comparison.csv", index=False)
    add_arm_comparison(aggregates, ["year"]).to_csv(OUT / "aggregate_level_comparison.csv", index=False)

    paths = [ROOT / "data" / "eurostat" / "env_wasgen_W121_TOTAL_HH.csv",
             ROOT / "data" / "eurostat" / "env_wasgen_W121_F.csv"]
    metadata = {
        "task": "bounded EU full-panel measurement-prior-location sensitivity",
        "scope": "two arms only; full EU27 panel; no rolling-origin or masked-state reruns",
        "data": {
            "files": [p.relative_to(ROOT).as_posix() for p in paths],
            "sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
            "observations": int(len(df)),
            "countries": int(C),
            "years": years,
            "empirical_F_TOTAL_ratio": {"n": 187, "median": 0.811, "p25": 0.623, "p75": 0.975,
                                        "note": "Descriptive panel ratio; the slope prior is not centered on this ratio."},
        },
        "model": {
            "only_changed_parameter": "logB_F prior location",
            "baseline_logB_loc": -0.2,
            "sensitivity_logB_loc": 0.0,
            "logB_scale": 0.3,
            "b_F_prior": "Normal(0, 0.5)",
            "all_other_model_settings_unchanged": True,
            "ratio_formula": "exp((exp(logB_F)-1)*theta_ref+b_F)",
            "common_theta_ref": common_theta_ref,
            "common_theta_ref_definition": "median of all baseline posterior conditional log-theta draws across member-wave cells",
            "empirical_ratio_is_not_B_or_coverage_slope": True,
        },
        "sampler": {
            "engine": "NumPyro NUTS",
            "chains": CHAINS,
            "chain_method": "sequential",
            "warmup_per_chain": WARMUP,
            "retained_draws_per_chain": SAMPLES_PER_CHAIN,
            "target_accept": TARGET_ACCEPT,
            "float64": bool(jax.config.x64_enabled),
            "jax_version": jax.__version__,
            "numpyro_version": numpyro.__version__,
            "numpy_version": np.__version__,
        },
        "runs": [{
            "arm": run["arm"], "logB_loc": run["logB_loc"], "seed": run["seed"],
            "state_draw_seed": run["state_draw_seed"], "runtime_seconds": run["runtime_seconds"],
            **run["diagnostics"],
        } for run in runs],
        "limits": [
            "Bounded sensitivity comparison, not a full refit/backtest robustness study.",
            "Posterior changes reflect both prior-location and finite MCMC Monte Carlo variability.",
            "Member and aggregate levels are posterior summaries within the stated model and anchor scope; they do not establish independent truth.",
        ],
        "output_files": sorted(output_names),
    }
    write_json(OUT / "run_metadata.json", metadata)
    print(ratios.to_string(index=False))


if __name__ == "__main__":
    main()
