"""Nature-style figure suite for the WM manuscript (plan4). v2.1 — all fixes in.

Produces paper/figs/fig1..fig6 (PDF, vector) + results/fig_audit/*.png (300 dpi).
Style: NPG palette, Arial, bold lowercase panel letters, 89/140/183 mm widths.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "paper" / "figs"
if not (ROOT / "paper").exists():          # clean reproducibility copy
    FIGS = ROOT / "results" / "figures"
AUD = ROOT / "results" / "fig_audit"
FIGS.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 140 * MM, 183 * MM
NPG = dict(blue="#3C5488", red="#E64B35", cyan="#4DBBD5", green="#00A087",
           salmon="#F39B7F", dusty="#8491B4", mint="#91D1C2", red2="#DC0000",
           brown="#7E6148", tan="#B09C85", ink="#2D3047", grey="#B9BDC6",
           gold="#F0C808")
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
SRC_STYLE = {
    "meescc": (NPG["red"], "s", "Official aggregate"),
    "mee_draft_2025": (NPG["red2"], "^", "Official broad caliber"),
    "gao_2025": (NPG["blue"], "o", "WGR series (Gao 2025)"),
    "wang_2022": (NPG["cyan"], "o", "Policy review (Wang 2023)"),
    "zheng_2017": (NPG["green"], "D", "Per-constr.-area (Zheng 2017)"),
    "envirunion_2017": (NPG["brown"], "v", "Industry assoc. 2017"),
    "cdw_alliance": (NPG["tan"], "v", "Industry assoc. 2014"),
}
SHORT = {"meescc": "Official agg.", "mee_draft_2025": "Broad caliber",
         "gao_2025": "WGR (Gao 2025)", "wang_2022": "Policy review",
         "zheng_2017": "Zheng 2017", "envirunion_2017": "Assoc. 2017",
         "cdw_alliance": "Assoc. 2014"}


def pan(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(AUD / f"{name}.png", dpi=300)
    plt.close(fig)
    print(name, "done")


def load_china_obs():
    df = pd.read_csv(ROOT / "data" / "china" / "china_observations_v01.csv")
    df["src"] = df["source_id"].replace(
        {"meescc_2024": "meescc", "meescc_2025": "meescc"})
    return df


# ================================================== F1 framework (183 mm)
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 92 * MM))
    ax.set_xlim(0, 183)
    ax.set_ylim(0, 92)
    ax.axis("off")

    def box(x, y, w, h, fc, ec="#3A3F4A", lw=0.7, r=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw))

    def arrow(x0, y0, x1, y1, color="#3A3F4A", lw=1.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=7, color=color, lw=lw,
                                     shrinkA=1, shrinkB=1))

    for x, w, lab in [(2, 52, "A  Contradictory evidence"),
                      (66, 50, "B  Coherent accounting model"),
                      (124, 57, "C  Decision-ready outputs")]:
        ax.text(x + w / 2, 88.5, lab, ha="center", fontsize=7.5,
                fontweight="bold", color=NPG["ink"])
        ax.plot([x, x + w], [86.8, 86.8], lw=0.8, color=NPG["ink"])

    fams = [("Official aggregates", "2.41 / 2.31 Gt (2023/24)", NPG["red"]),
            ("Official broad caliber", "\u201C> 3.0 Gt\u201D (2023)", NPG["red2"]),
            ("WGR series (Gao 2025)", "0.26 \u2192 2.28 Gt (2000\u201322)", NPG["blue"]),
            ("Stock-flow (Zheng 2017)", "2.36 Gt/yr (2003\u201313)", NPG["green"]),
            ("Policy review / associations", "2.4 Gt; 1.5\u20131.6 Gt", NPG["cyan"])]
    y0 = 66
    for lab, val, c in fams:
        box(2, y0, 52, 6.4, "#F4F6F8", ec=c, lw=0.9)
        ax.text(4.5, y0 + 4.2, lab, fontsize=6.5, va="center", color=NPG["ink"])
        ax.text(4.5, y0 + 1.4, val, fontsize=5.8, va="center",
                color="#5A5F6A", style="italic")
        y0 -= 8.6
    box(2, 8.5, 52, 9, "#ECEFF3")
    ax.text(28, 14.6, "same quantity, different calibers", ha="center",
            fontsize=6.2, style="italic", color=NPG["ink"])
    ax.text(28, 10.8, "2\u20137\u00D7 discrepancy in any given year", ha="center",
            fontsize=6.8, fontweight="bold", color=NPG["red2"])

    box(66, 62, 50, 20, "#EDF3F8", ec=NPG["blue"], lw=0.9)
    ax.text(91, 78.4, "anchor-referenced latent state  $\\theta_{c,t}$", ha="center", fontsize=7,
            fontweight="bold", color=NPG["blue"])
    ax.text(91, 74.3, "mean-reverting growth  (Eq. 2)", ha="center", fontsize=6)
    ax.text(91, 70.6, "nation-level latent state", ha="center",
            fontsize=6)
    xs = np.linspace(72, 110, 9)
    ax.plot(xs, [66.2] * 9, "-", color=NPG["blue"], lw=0.9, zorder=1)
    ax.scatter(xs, [66.2] * 9, s=8, color=NPG["blue"], zorder=2)
    ax.text(91, 63.4, "source coherence via shared latent state", ha="center",
            fontsize=5.6, color="#5A5F6A", style="italic")

    box(66, 44, 50, 13, "#FBF1EC", ec=NPG["red"], lw=0.9)
    ax.text(91, 53.6, "measurement model (Eq. 1)", ha="center", fontsize=6.6,
            fontweight="bold", color=NPG["red2"])
    ax.text(91, 49.6, r"$y_i = B_{m}\,\theta_{c(i),t(i)} + b_{m} + \varepsilon_i$",
            ha="center", fontsize=7)
    ax.text(91, 46.3, r"anchor: $B\!=\!1,\ b\!=\!0$  (statutory caliber)",
            ha="center", fontsize=5.8, color="#5A5F6A", style="italic")

    box(66, 27, 50, 12, "#EFF7F4", ec=NPG["green"], lw=0.9)
    ax.text(91, 35.0, "collapsed fully Bayesian inference", ha="center",
            fontsize=6.6, fontweight="bold", color=NPG["green"])
    ax.text(91, 30.8, r"NUTS on $\phi$; exact Gaussian $\theta\,|\,\phi,\mathbf{y}$",
            ha="center", fontsize=6.2)

    box(66, 8.5, 50, 12, "#F4F6F8")
    ax.text(91, 16.8, "parameter and state uncertainty", ha="center", fontsize=6.2,
            style="italic", color="#5A5F6A")
    ax.text(91, 12.0, "SBC-audited calibration", ha="center", fontsize=6.4,
            fontweight="bold", color=NPG["ink"])

    box(124, 58, 57, 24, "#EDF3F8", ec=NPG["blue"], lw=0.9)
    ax.text(152.5, 78.5, "anchor-referenced series + 90% interval", ha="center", fontsize=6.6,
            fontweight="bold", color=NPG["blue"])
    fy = np.linspace(128, 176, 30)
    ax.plot(fy, 62 + 8 * np.tanh((fy - 150) / 14), color=NPG["blue"], lw=1.1)
    ax.fill_between(fy, 60 + 8 * np.tanh((fy - 150) / 14),
                    64.5 + 8 * np.tanh((fy - 150) / 14),
                    color=NPG["blue"], alpha=0.22, lw=0)
    ax.scatter([132, 152, 170], [57.8, 66.0, 74.2], s=10, color=NPG["red"],
               marker="s", zorder=3)

    box(124, 33, 57, 19, "#FBF1EC", ec=NPG["red"], lw=0.9)
    ax.text(152.5, 48.5, "bias-factor table", ha="center", fontsize=6.6,
            fontweight="bold", color=NPG["red2"])
    rows = [("broad official caliber", "1.25\u00D7"), ("early WGR-type", "2.8\u00D7"),
            ("recent WGR series", "0.99\u00D7"), ("associations", "0.80\u00D7")]
    for k, (lab, v) in enumerate(rows):
        yy = 43.5 - 3.1 * k
        ax.text(127, yy, lab, fontsize=5.8, color=NPG["ink"])
        ax.text(178.5, yy, v, fontsize=5.8, ha="right", color=NPG["red2"])

    box(124, 8.5, 57, 19, "#EFF7F4", ec=NPG["green"], lw=0.9)
    ax.text(152.5, 23.5, "probabilistic forecasts", ha="center", fontsize=6.6,
            fontweight="bold", color=NPG["green"])
    ax.text(152.5, 18.6, "mean-reverting growth", ha="center", fontsize=6)
    ax.text(152.5, 14.6, "2.80 [1.49–4.91] Gt by 2030", ha="center",
            fontsize=6.2, color=NPG["ink"])

    arrow(54.8, 45, 65.2, 60)
    arrow(54.8, 20, 65.2, 32)
    arrow(91, 62, 91, 57.6)
    arrow(91, 44, 91, 39.6)
    arrow(91, 27, 91, 21.0)
    arrow(116.4, 68, 123.2, 70)
    arrow(116.4, 34, 123.2, 42)
    arrow(116.4, 18, 123.2, 18)
    save(fig, "fig1_framework")


# ================================================== F2 model DAG (140 mm)
def fig2():
    fig, ax = plt.subplots(figsize=(W15, 78 * MM))
    ax.set_xlim(0, 140)
    ax.set_ylim(19, 93)
    ax.axis("off")

    def edge(x0, y0, x1, y1, dashed=False, color="#3A3F4A", lw=0.7, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=6, color=color, lw=lw,
                                     linestyle=(0, (3, 2)) if dashed else "solid",
                                     shrinkA=3, shrinkB=3, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}"))

    def plate(x, y, w, h, label, lx, ly):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=2",
                                    fc="none", ec="#8A8F98", lw=0.8,
                                    linestyle=(0, (4, 2))))
        ax.text(lx, ly, label, fontsize=6.4, color="#5A5F6A", style="italic")

    plate(6, 29, 62, 57,
          "regions  $c = 1\\,\\dots\\,C$   \u00D7   waves  $t = 1\\,\\dots\\,T$",
          8, 88.5)
    ax.text(20, 82, r"$\kappa,\ \rho,\ \sigma_\eta$", ha="center", va="center",
            fontsize=7, zorder=4,
            bbox=dict(boxstyle="round,pad=0.35", fc="#EFF7F4", ec=NPG["green"],
                      lw=0.8))
    ax.text(6.5, 71.5, "growth dynamics", ha="left", fontsize=5.8,
            color="#5A5F6A", style="italic")
    chain = [(16, 58), (28, 51), (40, 44), (52, 37)]
    for k, (x, y) in enumerate(chain):
        ax.add_patch(Circle((x, y), 4.2, fc="#EDF3F8", ec=NPG["blue"], lw=0.8,
                            zorder=3))
        ax.text(x, y, rf"$\theta_{{c,{k+1}}}$", ha="center", va="center",
                fontsize=6.8, zorder=4)
    for (x0, y0), (x1, y1) in zip(chain[:-1], chain[1:]):
        edge(x0 + 3.2, y0 - 2.6, x1 - 3.2, y1 + 2.6)
    ax.text(36, 64.8, r"$g_t=\kappa+\rho\,(g_{t-1}-\kappa)+\eta_t$",
            ha="center", fontsize=6.4)
    edge(22, 79.5, 24, 66.8, dashed=True, color=NPG["green"])
    ax.text(6.5, 24.5, r"$\theta_{c,1}\!\sim\!\mathrm{N}(\mu_{0,c},\sigma_0^2)$",
            fontsize=6.2, color="#5A5F6A")

    plate(84, 20, 52, 66, "observations  $i = 1\\,\\dots\\,n$", 88, 88.5)
    ax.text(110, 80, r"$B_{m},\ b_{m},\ \sigma_{m}$", ha="center", va="center",
            fontsize=7, zorder=4,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FBF1EC", ec=NPG["red"],
                      lw=0.8))
    ax.text(110, 84.3, "caliber parameters", ha="center", fontsize=5.8,
            color="#5A5F6A", style="italic")
    ax.text(110, 62, r"$m(i)$", ha="center", va="center", fontsize=7, zorder=4,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8A8F98",
                      lw=0.8))
    ax.text(103, 62, "tradition / caliber", ha="right", fontsize=5.8,
            va="center", color="#5A5F6A", style="italic")
    ax.add_patch(Circle((110, 42), 5.0, fc="#D9DDE3", ec="#3A3F4A", lw=0.8,
                        zorder=3))
    ax.text(110, 42, r"$y_i$", ha="center", va="center", fontsize=7, zorder=4)
    ax.text(118.5, 42, "observed\nestimate", fontsize=5.8, va="center",
            color="#5A5F6A")
    ax.add_patch(FancyArrowPatch((56.2, 40), (104.5, 42), arrowstyle="-|>",
                                 mutation_scale=6, color=NPG["blue"], lw=0.8,
                                 linestyle=(0, (4, 2)), shrinkA=2, shrinkB=2,
                                 connectionstyle="arc3,rad=-0.08"))
    ax.text(80, 47.5, r"$\theta_{c(i),\,t(i)}$", fontsize=6.6, color=NPG["blue"])
    edge(110, 77.5, 110, 64.8, color=NPG["red"])
    edge(110, 59.5, 110, 47.6, color="#8A8F98")
    ax.add_patch(FancyBboxPatch((84, 22.5), 52, 9,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                fc="#FBF1EC", ec=NPG["red2"], lw=0.7))
    ax.text(110, 28.6, "identification: anchor caliber", ha="center",
            fontsize=5.9, color=NPG["red2"], fontweight="bold")
    ax.text(110, 25.1,
            r"$B\!=\!1,\ b\!=\!0$ " + "\u2014 all others relative to it",
            ha="center", fontsize=5.5, color="#5A5F6A", style="italic")
    save(fig, "fig2_dag")


# ================================================== F3 database overview
def fig3():
    obs = load_china_obs()
    nat = obs[(obs["level"] == "national") & (obs["caliber"] == "产生量")
              & (obs["substream"] == "建筑垃圾总量")]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 62 * MM),
                             gridspec_kw={"width_ratios": [1.25, 0.95, 0.8],
                                          "wspace": 0.55})
    ax = axes[0]
    used = set()
    for _, r in obs.iterrows():
        if r["src"] in SRC_STYLE:
            c, mk, lab = SRC_STYLE[r["src"]]
            lab_use = lab if r["src"] not in used else None
            used.add(r["src"])
        else:
            c, mk, lab = NPG["grey"], "o", "Sub-national records"
            lab_use = lab if "sub" not in used else None
            used.add("sub")
        ax.scatter(r["year"], r["value_wt"],
                   s=22 if r["level"] == "national" else 12, color=c, marker=mk,
                   alpha=0.85, linewidths=0.4, edgecolors="white",
                   label=lab_use, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([100, 1000, 10000, 100000, 300000])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "3"])
    ax.set_ylim(15, 7e5)
    ax.set_ylabel("Reported value (Gt)")
    ax.set_xlabel("Year")
    ax.set_xlim(1999, 2027)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
              fontsize=5.4, handletextpad=0.4, ncol=3)
    ax.set_title("57 records (53 with mass), 19 sources", fontsize=7, pad=3,
                 loc="left")
    pan(ax, "a")

    ax = axes[1]
    subs = [("Excavated spoil", 176, 222), ("Demolition waste", 10.5, 200),
            ("Engineering waste", 7.4, 197), ("Decoration waste", 5.4, 201),
            ("Slurry", 4.7, 166)]
    y = np.arange(len(subs))[::-1]
    ax.barh(y, [s[1] for s in subs], height=0.62,
            color=[NPG["blue"], NPG["cyan"], NPG["dusty"], NPG["mint"],
                   NPG["salmon"]])
    for yi, (lab, v, nc) in zip(y, subs):
        ax.text(v + 0.4, yi, f"{v:.1f} ({nc})", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in subs])
    ax.set_xlabel("2024 sub-stream generation (10 Mt); (n) = reporting cities")
    ax.set_xlim(0, 230)
    ax.set_title("Statutory sub-streams sum to\n2.04 Gt = 0.88\u00D7 anchor",
                 fontsize=7, pad=3, loc="left")
    pan(ax, "b")

    ax = axes[2]
    fams = ["meescc", "mee_draft_2025", "gao_2025", "wang_2022", "zheng_2017",
            "envirunion_2017", "cdw_alliance"]
    labs = [SHORT[f] for f in fams]
    natc = [len(nat[nat["src"] == f]) for f in fams]
    othc = [len(obs[obs["src"] == f]) - natc[k] for k, f in enumerate(fams)]
    y = np.arange(len(fams))[::-1]
    ax.barh(y + 0.19, natc, height=0.36, color=NPG["blue"], label="national")
    ax.barh(y - 0.19, othc, height=0.36, color=NPG["gold"],
            label="sub-national")
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("Observations")
    ax.legend(frameon=False, loc="lower right", fontsize=5.6)
    ax.set_title("National vs. sub-national counts",
                 fontsize=7, pad=3, loc="left")
    pan(ax, "c")
    save(fig, "fig3_database")


# ================================================== F4 EU validation
def fig4():
    e3 = pd.read_csv(RES / "e3_hierarchy_check.csv")
    tot = pd.read_csv(ROOT / "data" / "eurostat" / "env_wasgen_W121_TOTAL_HH.csv")
    f = pd.read_csv(ROOT / "data" / "eurostat" / "env_wasgen_W121_F.csv")
    EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI",
            "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL",
            "PT", "RO", "SE", "SI", "SK"]
    tot = tot[tot["geo"].isin(EU27) & (tot["value"] > 0)]
    f = f[f["geo"].isin(EU27) & (f["value"] > 0)]
    m = tot.merge(f, on=["geo", "time"], suffixes=("_t", "_f"))
    m["lr"] = np.log(m["value_f"] / m["value_t"])
    fig, axes = plt.subplots(1, 4, figsize=(W2, 46 * MM),
                             gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    ax.fill_between(e3["year"], e3["lo"], e3["hi"], color=NPG["blue"],
                    alpha=0.22, lw=0, label="90% band")
    ax.plot(e3["year"], e3["est_Mt"], color=NPG["blue"], lw=1.1, marker="o",
            ms=2.4, label="posterior sum")
    ax.plot(e3["year"], e3["truth_Mt"], color=NPG["red"], lw=1.1, ls="--",
            marker="s", ms=2.4, label="official EU-27")
    ax.set_ylabel("Mineral CDW (Mt)")
    ax.set_xticks([2010, 2014, 2018, 2022])
    ax.legend(frameon=False, loc="upper left", fontsize=5.4)
    pan(ax, "a", dy=1.04)

    ax = axes[1]
    ax.hist(m["lr"], bins=26, color=NPG["dusty"], alpha=0.85, density=True)
    ax.axvline(m["lr"].median(), color=NPG["red"], ls="--", lw=1.0)
    r_model = -1.616 + (1.085 - 1.0) * float(np.median(np.log(tot["value"])))
    ax.axvline(r_model, color=NPG["blue"], lw=1.2)
    ax.set_ylim(0, 2.9)
    ax.text(-4.45, 2.62, "empirical median (0.81\u00D7)", fontsize=5.6,
            color=NPG["red"])
    ax.text(-4.45, 2.36, "model, median level (0.65\u00D7)", fontsize=5.6,
            color=NPG["blue"])
    ax.set_xlabel("log( F / total caliber )")
    ax.set_ylabel("Density")
    pan(ax, "b", dy=1.04)

    ax = axes[2]
    e2 = pd.concat([pd.read_csv(RES / "e2_mask_recovery_IE.csv").assign(g="IE"),
                    pd.read_csv(RES / "e2_mask_recovery_DE.csv").assign(g="DE")])
    x = np.arange(4)
    hi = e2["hi"].to_numpy()
    lo = e2["lo"].to_numpy()
    med = e2["pred_Mt"].to_numpy()
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", ms=3.4,
                color=NPG["blue"], capsize=2.4, lw=0.9, label="posterior (90% PI)")
    ax.scatter(x, e2["truth_Mt"], marker="s", s=16, color=NPG["red"], zorder=3,
               label="truth")
    ax.scatter(x, e2["truth_Mt"] * (1 + e2["persistence_err_pct"] / 100),
               marker="^", s=16, color=NPG["grey"], zorder=3,
               label="persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(["IE\n2020", "IE\n2022", "DE\n2020", "DE\n2022"])
    ax.set_ylabel("Generation (Mt)")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 700)
    ax.legend(frameon=False, loc="center right", fontsize=5.4)
    pan(ax, "c", dy=1.04)

    ax = axes[3]
    cov = []
    for tag in ["W1", "W2"]:
        fw = pd.read_csv(RES / f"eu_forecast_{tag}.csv")
        cov.append((tag, fw["covered"].mean(), len(fw)))
    x = np.arange(2)
    ax.bar(x, [c[1] for c in cov], width=0.5, color=[NPG["cyan"], NPG["green"]])
    for xi, c in zip(x, cov):
        ax.text(xi, c[1] + 0.015, f"{c[1]:.2f}", ha="center", fontsize=6.5)
        ax.text(xi, 0.05, f"n = {c[2]}", ha="center", fontsize=5.6,
                color="white")
    ax.axhline(0.90, color=NPG["red"], ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["train \u22642018\n\u2192 2020", "train \u22642020\n\u2192 2022"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("90% PI coverage")
    pan(ax, "d", dy=1.04)
    save(fig, "fig4_eu")


# ================================================== F5 China series (140 mm)
def fig5():
    post = pd.read_csv(RES / "china_national_posterior_v2c.csv")
    fc = pd.read_csv(RES / "china_forecast_v2c.csv")
    nat = load_china_obs()
    nat = nat[(nat["level"] == "national") & (nat["caliber"] == "产生量")
              & (nat["substream"] == "建筑垃圾总量")]
    fig, ax = plt.subplots(figsize=(W15, 62 * MM))
    Y = 1e5
    ax.fill_between(post["year"], post["lo90_wt"] / Y, post["hi90_wt"] / Y,
                    color=NPG["blue"], alpha=0.20, lw=0,
                    label="coherent estimate (90% PI)")
    ax.plot(post["year"], post["median_wt"] / Y, color=NPG["blue"], lw=1.4,
            label="coherent estimate (median)")
    ax.fill_between(fc["year"], fc["lo90_wt"] / Y, fc["hi90_wt"] / Y,
                    color=NPG["cyan"], alpha=0.18, lw=0)
    ax.plot(fc["year"], fc["median_wt"] / Y, color=NPG["cyan"], lw=1.3,
            ls="--", label="forecast (mean-reverting)")
    for src, g in nat.groupby("src"):
        c, mk, lab = SRC_STYLE.get(src, (NPG["grey"], "o", src))
        ax.scatter(g["year"], g["value_wt"] / Y, s=22, color=c, marker=mk,
                   linewidths=0.4, edgecolors="white", label=lab, zorder=3)
    ax.set_ylabel("National CDW generation (Gt)")
    ax.set_xlabel("Year")
    ax.set_xlim(1999, 2031)
    ax.set_ylim(0, 5.2)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=6,
              handletextpad=0.4)
    fig.tight_layout()
    save(fig, "fig5_china")


# ================================================== F6 bias forest (89 mm)
def fig6():
    sb = pd.read_csv(RES / "china_source_bias_v2c.csv")
    sb = sb[~sb["anchor"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(W1, 52 * MM))
    y = np.arange(len(sb))
    for yi, (_, r) in zip(y, sb.iterrows()):
        ax.plot([r["ratio_lo"], r["ratio_hi"]], [yi, yi], color=NPG["blue"],
                lw=1.8, solid_capstyle="round", alpha=0.6)
        ax.scatter([r["ratio_at_mean"]], [yi], s=22, color=NPG["blue"],
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.text(max(r["ratio_hi"] * 1.05, r["ratio_at_mean"] + 0.15), yi,
                f"{r['ratio_at_mean']:.2f}\u00D7", va="center", fontsize=6)
    ax.axvline(1.0, color=NPG["red"], ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([SRC_STYLE.get(s, (0, 0, s))[2] for s in sb["source"]])
    ax.set_xscale("log")
    ax.set_xticks([0.6, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.6", "1", "1.5", "2", "3"])
    ax.set_xlim(0.45, 4.8)
    ax.set_xlabel("Level ratio vs. anchor caliber (90% CI)")
    fig.tight_layout()
    save(fig, "fig6_bias")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
