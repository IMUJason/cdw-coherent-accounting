"""Independent R5 verification; never overwrites archived v2c fit outputs.

Run using NumPyro 0.19.0/JAX 0.6.2, with JAX double precision enabled.
Each China check uses two independent sequential chains. Sensitivities change
one assumption at a time. Files are written only under results/review_r5/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
import pandas as pd
from scipy import stats

import w3_china_national_v2c as cn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_r5"
OUT.mkdir(exist_ok=True)


def algebra():
    """Compare precision algebra with direct covariance conditioning."""
    rng = np.random.default_rng(20261019)
    errors = []
    for n in [1, 2, 4, 24]:
        for rho in [-0.7, 0.0, 0.8, 0.97]:
            s = 0.12
            lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            K = s**2 / (1-rho**2) * rho**lag
            err = np.max(np.abs(cn.ar1_prec(n, rho, s) @ K - np.eye(n)))
            errors.append(float(err))
    df = cn.load_obs()
    phi = dict(mu0=np.log(26279.), c=.02, rho=.8, sd_eta=.08,
               logB=np.array([0., .01, -.01, .02, -.03, .01, -.02]),
               b=np.array([0., -.12, .2, .4, -.2, .3, -.1]),
               log_sd_m=np.log(np.array([.05, .07, .15, .15, .15, .15, .15])))
    m, P = cn.theta_prior(phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"], cn.T)
    K = np.linalg.inv(P)
    H = np.zeros((len(df), cn.T))
    H[np.arange(len(df)), df.t.to_numpy()] = np.exp(phi["logB"])[df.m]
    b = phi["b"][df.m]
    R = np.diag(np.exp(2*phi["log_sd_m"])[df.m])
    V = H @ K @ H.T + R
    direct_m = m + K @ H.T @ np.linalg.solve(V, df.y.to_numpy()-H@m-b)
    direct_K = K-K@H.T@np.linalg.solve(V, H@K)
    cond_m, cond_K = cn.theta_posterior(phi, df, cn.T)
    state = rng.multivariate_normal(m, K)
    obs = H@state+b+rng.multivariate_normal(np.zeros(len(df)), R)
    # Gaussian integral identity evaluated at any theta, independent of data fit.
    log_marg = stats.multivariate_normal.logpdf(obs, H@m+b, V)
    post_m = m+K@H.T@np.linalg.solve(V, obs-H@m-b)
    identity = (stats.multivariate_normal.logpdf(state, m, K)
                +stats.multivariate_normal.logpdf(obs, H@state+b, R)
                -stats.multivariate_normal.logpdf(state, post_m, direct_K))
    report = dict(ar_precision_max_error=max(errors),
                  conditional_mean_max_error=float(np.max(np.abs(direct_m-cond_m))),
                  conditional_cov_max_error=float(np.max(np.abs(direct_K-cond_K))),
                  log_marginal_identity_error=float(abs(log_marg-identity)))
    assert max(errors) < 1e-9
    assert report["conditional_mean_max_error"] < 1e-8
    assert report["conditional_cov_max_error"] < 1e-8
    assert report["log_marginal_identity_error"] < 1e-7
    (OUT/"algebra_checks.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def conditional(phi, df, corr):
    if not corr:
        return cn.theta_posterior(phi, df, cn.T)
    m, P = cn.theta_prior(phi["mu0"], phi["c"], phi["rho"], phi["sd_eta"], cn.T)
    mi, ti = df.m.to_numpy(), df.t.to_numpy()
    H = np.zeros((len(df), cn.T))
    H[np.arange(len(df)), ti] = np.exp(phi["logB"])[mi]
    sd = np.exp(phi["log_sd_m"])[mi]
    lag = np.abs(ti[:, None]-ti[None, :])
    extra = np.where((mi[:, None] == 1) & (mi[None, :] == 1) & (lag > 0), corr**lag, 0.)
    R = sd[:, None]*sd[None, :]*(np.eye(len(df))+extra)
    cov = np.linalg.inv(P+H.T@np.linalg.solve(R, H))
    mean = cov@(P@m+H.T@np.linalg.solve(R, df.y.to_numpy()-phi["b"][mi]))
    return mean, (cov+cov.T)/2


def fit_china(case, warmup=1200, samples=1500):
    started = time.monotonic()
    df = cn.load_obs()
    kwargs = {}
    if case == "correlated":
        kwargs["gao_corr"] = .7
    elif case == "omit_bounds":
        df = df[~df.src.isin(["mee_draft_2025", "cdw_alliance"])].copy()
    elif case == "unit_slopes":
        kwargs["unit_slopes"] = True
    elif case == "prior_wide":
        kwargs["mu0_scale"] = 1.8
    elif case != "base":
        raise ValueError(case)
    model = cn.make_model(df, cn.T, **kwargs)
    mc = MCMC(NUTS(model, target_accept_prob=.9), num_warmup=warmup,
              num_samples=samples, num_chains=2, chain_method="sequential",
              progress_bar=False)
    seed = 20261019 + ["base", "correlated", "omit_bounds", "unit_slopes", "prior_wide"].index(case)
    print("START", case, "n", len(df), "2 chains", flush=True)
    mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    chain_samples = mc.get_samples(group_by_chain=True)
    diag_rows = []
    for name, values in summary(chain_samples, group_by_chain=True).items():
        for idx in np.ndindex(np.asarray(values["mean"]).shape):
            label = name + ("["+",".join(map(str, idx))+"]" if idx else "")
            diag_rows.append(dict(parameter=label, **{k:float(np.asarray(v)[idx]) for k,v in values.items()}))
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(OUT/f"china_{case}_diagnostics.csv", index=False)
    smp = {k:np.asarray(v) for k,v in mc.get_samples().items()}
    np.savez_compressed(OUT/f"china_{case}_hyper.npz", **chain_samples)
    take = np.arange(0, 2*samples, 2)
    rng = np.random.default_rng(seed+100)
    states = np.empty((cn.T, len(take)))
    obs_pred = np.empty(len(take))
    ratios = np.empty((len(cn.FIT_SOURCES), len(take)))
    fut = np.empty((6, len(take)*20))
    for si, s in enumerate(take):
        phi = dict(mu0=float(smp["mu0"][s]), c=float(smp["c"][s]),
                   rho=float(smp["rho"][s]), sd_eta=float(smp["sd_eta"][s]),
                   logB=np.r_[0., smp["logB_f"][s]], b=np.r_[0., smp["b_f"][s]],
                   log_sd_m=np.r_[smp["log_sd_m_free"][s, :1], np.log(.07), smp["log_sd_m_free"][s, 1:]])
        cm, ck = conditional(phi, df, kwargs.get("gao_corr", 0.))
        states[:, si] = rng.multivariate_normal(cm, ck, check_valid="raise", tol=1e-8)
        obs_pred[si] = states[-1, si]+rng.normal(0., np.exp(phi["log_sd_m"][0]))
        ratios[:, si] = np.exp((np.exp(phi["logB"])-1)*states[:, si].mean()+phi["b"])
        g = np.full(20, states[-1, si]-states[-2, si])
        th = np.full(20, states[-1, si])
        for h in range(6):
            g = phi["c"]+phi["rho"]*(g-phi["c"])+rng.normal(0., phi["sd_eta"],20)
            th = th+g
            fut[h, si*20:(si+1)*20] = np.exp(th)
    np.save(OUT/f"china_{case}_theta.npy", states)
    np.save(OUT/f"china_{case}_forecast.npy", fut)
    q=np.quantile(np.exp(states)/1e5,[.5,.05,.95],axis=1)
    frame=pd.DataFrame(dict(year=cn.YEARS,median_Gt=q[0],lo90_Gt=q[1],hi90_Gt=q[2]))
    frame.to_csv(OUT/f"china_{case}_posterior.csv",index=False)
    qf=np.quantile(fut/1e5,[.5,.05,.95],axis=1)
    pd.DataFrame(dict(year=np.arange(2025,2031),median_Gt=qf[0],lo90_Gt=qf[1],hi90_Gt=qf[2])).to_csv(OUT/f"china_{case}_forecast.csv",index=False)
    br=np.quantile(ratios,[.5,.05,.95],axis=1)
    pd.DataFrame(dict(source=cn.FIT_SOURCES,ratio=br[0],lo90=br[1],hi90=br[2])).to_csv(OUT/f"china_{case}_bias.csv",index=False)
    # Unit-slopes contains deterministic zeros: exclude undefined diagnostics there.
    finite_rhat=diagnostics.r_hat[np.isfinite(diagnostics.r_hat)]
    finite_ess=diagnostics.n_eff[np.isfinite(diagnostics.n_eff)]
    report=dict(case=case,observations=len(df),chains=2,warmup=warmup,samples_per_chain=samples,
                seed=seed,divergences=int(np.asarray(mc.get_extra_fields()["diverging"]).sum()),
                max_rhat=float(finite_rhat.max()),min_ess=float(finite_ess.min()),
                seconds=time.monotonic()-started,year2023=q[:,23].tolist(),
                year2024=q[:,24].tolist(),year2030=qf[:,-1].tolist(),
                obs2024_interval_Gt=np.quantile(np.exp(obs_pred)/1e5,[.05,.95]).tolist(),
                jax=jax.__version__,numpyro=numpyro.__version__,x64=bool(jax.config.x64_enabled))
    (OUT/f"china_{case}_run.json").write_text(json.dumps(report,indent=2))
    print("DONE",report,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("case",choices=["algebra","base","correlated","omit_bounds","unit_slopes","prior_wide","all"])
    args=ap.parse_args()
    if args.case in ["algebra","all"]:
        algebra()
    if args.case == "all":
        for case in ["base","correlated","omit_bounds","unit_slopes","prior_wide"]:
            fit_china(case)
    elif args.case != "algebra":
        fit_china(args.case)


if __name__ == "__main__":
    main()
