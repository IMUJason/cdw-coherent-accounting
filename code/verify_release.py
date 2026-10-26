"""Offline release verification for the minimal reproducibility package.

Checks data invariants, recomputes released summaries from released draws
and CSVs, and scans text artifacts for private absolute paths. No MCMC.
Run from the package root:  python code/verify_release.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
checks: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    obs = pd.read_csv(ROOT / "data" / "china" / "china_observations_v01.csv")
    src_map = {"meescc_2024": "meescc", "meescc_2025": "meescc"}
    obs["src"] = obs["source_id"].replace(src_map)
    fit_sources = {"meescc", "gao_2025", "wang_2022", "zheng_2017",
                   "envirunion_2017", "mee_draft_2025", "cdw_alliance"}
    sel = obs[(obs["level"] == "national") & (obs["caliber"] == "产生量")
              & (obs["substream"] == "建筑垃圾总量") & (obs["src"].isin(fit_sources))]
    sel = sel[(sel["year"] >= 2000) & (sel["year"] <= 2024)]
    record("china_obs_records", len(obs) == 58, f"records={len(obs)} (expect 58)")
    record("china_obs_sources", obs["source_id"].nunique() == 19,
           f"sources={obs['source_id'].nunique()} (expect 19)")
    record("china_obs_numeric", obs["value_wt"].notna().sum() == 54,
           f"numeric={int(obs['value_wt'].notna().sum())} (expect 54)")
    record("china_fit_set", len(sel) == 30 and sel["src"].nunique() == 7,
           f"fit obs={len(sel)}, processes={sel['src'].nunique()} (expect 30/7)")

    summary = json.loads((RES / "review_r7" / "sbc_main_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(RES / "review_r7" / "sbc_main.csv")
    record("sbc_replicates", len(frame) == 32 and frame["rep"].nunique() == 32,
           f"rows={len(frame)}")
    for arm in ["oracle", "em", "emreg", "map", "full"]:
        recomputed = float(frame[f"{arm}_cov"].mean())
        released = float(summary[arm]["coverage"])
        record(f"sbc_{arm}_coverage",
               abs(recomputed - released) < 1e-9,
               f"recomputed={recomputed:.6f} released={released:.6f}")

    fut = np.load(RES / "china_forecast_draws_v2c.npy")
    Gy = fut[1:6, :] / 1e5
    # illustrative scenario values for the capacity example; not a claim
    # about any specific facility
    capex, cap_t = 4186.76e4, 17.52e4
    crf = 0.08 * 1.08 ** 20 / (1.08 ** 20 - 1)
    k = capex / cap_t * crf
    p, h = 95.0, 25.0
    target = (p - k) / (p + h)
    c_star = float(np.quantile(Gy.reshape(-1), target))
    c_point = float(np.quantile(Gy.reshape(-1), 0.5))
    meta = json.loads((RES / "decision_value_annual_meta.json").read_text(encoding="utf-8"))
    record("decision_c_star", abs(c_star - meta["C_star"]) < 1e-3,
           f"recomputed={c_star:.4f} released={meta['C_star']:.4f}")
    record("decision_c_point", abs(c_point - meta["C_point"]) < 1e-3,
           f"recomputed={c_point:.4f} released={meta['C_point']:.4f}")
    sens = pd.read_csv(RES / "decision_value_annual_sensitivity.csv")
    record("decision_grid",
           sorted(sens["c_pen"].unique().tolist()) == [30.0, 37.0, 70.0, 95.0, 120.0, 140.0, 200.0]
           and sorted(sens["c_idle"].unique().tolist()) == [2.5, 25.0, 45.0],
           f"p={sorted(sens['c_pen'].unique().tolist())}, h={sorted(sens['c_idle'].unique().tolist())}")

    hold = pd.read_csv(RES / "external_consistency_r7.csv")
    latent = hold[hold["test"] == "A_holdout_2024_latent"].iloc[0]
    report_row = hold[hold["test"] == "A_holdout_2024_observation"].iloc[0]
    record("holdout_latent_covers", bool(latent["covered"])
           and abs(latent["median_Gt"] - 2.4971) < 0.01,
           f"median={latent['median_Gt']:.4f}, interval=[{latent['lo90_Gt']:.3f},{latent['hi90_Gt']:.3f}]")
    record("holdout_report_wider", bool(report_row["covered"])
           and report_row["lo90_Gt"] < latent["lo90_Gt"]
           and report_row["hi90_Gt"] > latent["hi90_Gt"],
           f"report interval=[{report_row['lo90_Gt']:.3f},{report_row['hi90_Gt']:.3f}]")

    prior = pd.read_csv(RES / "review_r7" / "prior" / "base_source_ratios.csv")
    zheng = prior[prior["source"] == "zheng_2017"].iloc[0]
    record("prior_theta_ref", abs(float(zheng["theta_ref"]) - 11.6402) < 1e-3,
           f"theta_ref={zheng['theta_ref']:.4f}")
    record("prior_zheng_ratio", abs(float(zheng["ratio_median"]) - 2.808) < 0.01,
           f"zheng base median={zheng['ratio_median']:.4f}")

    eu_ratio = pd.read_csv(RES / "review_r7" / "eu_prior" / "source_ratio_at_common_theta.csv")
    base_r = eu_ratio[eu_ratio["arm"] == "baseline"].iloc[0]["ratio_median"]
    loc0_r = eu_ratio[eu_ratio["arm"] == "loc0"].iloc[0]["ratio_median"]
    record("eu_prior_two_arms", len(eu_ratio) == 2 and abs(loc0_r / base_r - 1) < 0.01,
           f"baseline={base_r:.4f}, loc0={loc0_r:.4f}")

    loso = pd.read_csv(RES / "review_r7" / "loso" / "loso_summary.csv")
    single = loso[loso["dropped_obs"] == 1]
    anchor_row = loso[loso["dropped"] == "meescc"].iloc[0]
    record("loso_single_point_stable",
           len(loso) == 7 and (single["y2023_median"] - 2.36).abs().max() < 0.02,
           f"rows={len(loso)}, single-drop 2023 range="
           f"[{single['y2023_median'].min():.3f},{single['y2023_median'].max():.3f}]")
    record("loso_anchor_identifies",
           anchor_row["y2023_hi"] - anchor_row["y2023_lo"] > 8.0,
           f"drop-anchor 2023 interval=[{anchor_row['y2023_lo']:.2f},{anchor_row['y2023_hi']:.2f}]")

    cens = pd.read_csv(RES / "review_r7" / "censored" / "censored_summary.csv")
    c23 = cens[(cens["scope"] == "posterior") & (cens["year"] == 2023)].iloc[0]
    c30 = cens[(cens["scope"] == "forecast") & (cens["year"] == 2030)].iloc[0]
    record("censored_close_to_production",
           abs(c23["median"] - 2.365) < 0.02 and abs(c30["median"] - 2.772) < 0.05,
           f"2023={c23['median']:.3f}, 2030={c30['median']:.3f}")

    srcs = pd.read_csv(ROOT / "data" / "china" / "china_sources_v01.csv")
    obs_counts = obs.groupby("source_id").size()
    src_counts = srcs.groupby("source_id")["n_obs"].sum()
    record("sources_table_consistent",
           set(obs["source_id"]).issubset(set(srcs["source_id"]))
           and int(srcs["n_obs"].sum()) == 58
           and all(src_counts[sid] == n for sid, n in obs_counts.items()),
           f"n_obs total={int(srcs['n_obs'].sum())}, per-source match="
           f"{all(src_counts[sid] == n for sid, n in obs_counts.items())}")

    drift = pd.read_csv(RES / "china_forecast.csv")
    d2030 = float(drift[drift["year"] == 2030]["median_wt"].iloc[0]) / 1e5
    record("constant_drift_2030", abs(d2030 - 4.12) < 0.01,
           f"2030 median={d2030:.3f} Gt")

    noise_run = json.loads((RES / "review_r15_noise_prior" / "noise_wide_run.json")
                           .read_text(encoding="utf-8"))
    record("noise_arm_prior_setting",
           noise_run["prior_parameters"]["log_sd_prior_sd"] == 0.8,
           f"log_sd_prior_sd={noise_run['prior_parameters']['log_sd_prior_sd']}")

    noise = json.loads((RES / "review_r15_noise_prior" / "noise_prior_summary.json")
                       .read_text(encoding="utf-8"))
    record("noise_prior_endpoint_sensitive",
           abs(noise["max_abs_endpoint_pct_change"] - 15.53) < 0.05
           and noise["max_abs_median_pct_change"] < 1.2
           and noise["run"]["divergences_total"] == 0,
           f"max median change={noise['max_abs_median_pct_change']:.2f}%, "
           f"max endpoint change={noise['max_abs_endpoint_pct_change']:.2f}%")

    dyn = pd.read_csv(RES / "review_r7" / "dynprior" / "dynprior_summary.csv")
    base30 = dyn[dyn["arm"] == "base"].iloc[0]["y2030_median"]
    sig30 = dyn[dyn["arm"] == "sigma_wide"].iloc[0]["y2030_median"]
    kap30 = dyn[dyn["arm"] == "kappa_wide"].iloc[0]["y2030_median"]
    record("dynprior_expected",
           len(dyn) == 3 and abs(sig30 - base30) / base30 < 0.01 and kap30 > base30,
           f"base={base30:.2f}, sigma_wide={sig30:.2f}, kappa_wide={kap30:.2f}")

    post = pd.read_csv(RES / "china_national_posterior_v2c.csv")
    g2023 = float(post[post["year"] == 2023]["median_wt"].iloc[0]) / 1e5
    record("unit_conversion_2023", abs(g2023 - 2.36) < 0.01,
           f"2023 median={g2023:.4f} Gt (raw wt={float(post[post['year']==2023]['median_wt'].iloc[0]):.0f})")

    drive = chr(58) + chr(92) * 2                   # ':' + regex-escaped backslash
    private_patterns = [
        "[A-Z]" + drive,                            # C:\ style absolute paths
        "/Us" + "ers/",
        "\u6211\u7684\u575a\u679c\u4e91", "DE" + "LL",
        "ana" + "conda3"]
    leaks = []
    for pattern in private_patterns:
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json", ".txt", ".csv", ".log"}:
                continue
            if "verify_release" in path.name:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if re.search(pattern, text):
                leaks.append(f"{path.relative_to(ROOT).as_posix()}~{pattern}")
    record("no_private_paths", not leaks, f"leaks={leaks[:5] if leaks else 'none'}")

    report = {"checks": checks,
              "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks)}
    out = RES / "review_r7" / "release_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
