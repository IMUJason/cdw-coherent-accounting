"""Figure 7: external consistency and decision value (two panels)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "paper" / "figs"
if not (ROOT / "paper").exists():          # clean reproducibility copy
    FIGS = ROOT / "results" / "figures"
AUD = ROOT / "results" / "fig_audit"
FIGS.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
NPG = dict(blue="#3C5488", red="#E64B35", cyan="#4DBBD5", green="#00A087",
           dusty="#8491B4", ink="#2D3047", grey="#B9BDC6", gold="#F0C808")
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

post = pd.read_csv(RES / "china_national_posterior_v2c.csv")
ext = pd.read_csv(RES / "external_consistency.csv")
dv = pd.read_csv(RES / "decision_value_annual.csv")
fut = np.load(RES / "china_forecast_draws_v2c.npy")
Gy = fut[1:6, :] / 1e5                      # Gt/yr, years 2026-2030

fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 55 * MM),
                         gridspec_kw={"wspace": 0.30, "width_ratios": [1, 1.1]})

# ---------------- (a) external consistency ----------------
ax = axes[0]
rows = {int(r.year): r for _, r in post.iterrows()}
xs, meds, los, his = [], [], [], []
for k, yr in enumerate([2022, 2023, 2024]):
    r = rows[yr]
    xs.append(k - 1)
    meds.append(r.median_wt / 1e5)
    los.append(r.lo90_wt / 1e5)
    his.append(r.hi90_wt / 1e5)
ax.errorbar(xs, meds, yerr=[np.array(meds) - los, np.array(his) - meds],
            fmt="o", ms=4, color=NPG["blue"], capsize=3, lw=1.0,
            label="posterior (90% interval)")
# external points
ax.scatter([-0.24], [2.39], marker="^", s=34, color=NPG["red"], zorder=4,
           label="Gao 2023 forecast (same source)")
ax.scatter([0.76], [2.31], marker="s", s=30, color=NPG["green"], zorder=4,
           label="2024 anchor, withheld")
# cross-tradition market estimate at 2022
ax.scatter([-1.24], [2.75], marker="*", s=52, color=NPG["gold"],
           edgecolors="#5A5F6A", linewidths=0.4, zorder=4,
           label="2022 market estimate (NBS-lineage)")
# hold-out refit result
ho = ext[ext.test == "A_holdout_2024"].iloc[0]
ax.errorbar([1.28], [ho.median_Gt],
            yerr=[[ho.median_Gt - ho.lo_Gt], [ho.hi_Gt - ho.median_Gt]],
            fmt="D", ms=3.6, mfc="white", mec=NPG["dusty"], ecolor=NPG["dusty"],
            capsize=3, lw=0.9, label="2024 refit (anchor withheld)")
ax.set_xticks([-1, 0, 1, 1.46])
ax.set_xticklabels(["2022", "2023", "2024", "2024\n(refit)"])
ax.set_xlim(-1.5, 1.75)
ax.set_ylabel("National CDW generation (Gt)")
ax.set_ylim(1.95, 3.58)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none",
          loc="upper left", fontsize=5.0, ncol=1,
          handletextpad=0.5, borderaxespad=0.3)
ax.set_title("Consistency checks", loc="left")
ax.text(-0.20, 1.05, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

# ---------------- (b) decision value ----------------
ax = axes[1]
k_cap, c_pen, c_idle = 24.3397, 95.0, 25.0   # Tuoketuo EIA, 8%/20 yr
grid = np.linspace(1.6, 4.0, 400)
curves = []
for C in grid:
    annual = (k_cap * C
              + c_pen * np.maximum(Gy - C, 0).mean(axis=1).mean()
              + c_idle * np.maximum(C - Gy, 0).mean(axis=1).mean())
    curves.append(annual / Gy.mean())
curves = np.array(curves)
ax.plot(grid, curves, color=NPG["ink"], lw=1.2)
C_point = float(dv.loc[dv.policy.str.startswith("point"), "C_Gt"].iloc[0])
C_bayes = float(dv.loc[dv.policy.str.startswith("Bayes"), "C_Gt"].iloc[0])
c_point = float(dv.loc[dv.policy.str.startswith("point"), "cost_per_t"].iloc[0])
c_bayes = float(dv.loc[dv.policy.str.startswith("Bayes"), "cost_per_t"].iloc[0])
ax.scatter([C_point], [c_point], s=40, color=NPG["red"], zorder=4)
ax.scatter([C_bayes], [c_bayes], s=40, color=NPG["green"], zorder=4)
ax.text((C_point + C_bayes) / 2, 36.42,
        f"{C_bayes - C_point:+.2f} Gt ({(C_bayes - C_point) / C_point * 100:+.1f}%)",
        ha="center", va="top", fontsize=5.8, color="#5A5F6A")
ax.set_xlabel("Committed processing capacity C (Gt/yr)")
ax.set_ylabel("Expected cost (CNY/t generated)", labelpad=2)
ax.set_ylim(36.30, 38.45)
ax.set_title("Capacity planning under asymmetric loss", loc="left")
ax.text(-0.17, 1.05, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

fig.savefig(FIGS / "fig7_external_decision.pdf")
fig.savefig(AUD / "fig7_external_decision.png", dpi=300)
plt.close(fig)
print("fig7 done; C_point =", round(C_point, 3), "C_bayes =", round(C_bayes, 3),
      "| median of mixture =", round(float(np.median(Gy.reshape(-1))), 3),
      "| min cost =", round(float(curves.min()), 3))
