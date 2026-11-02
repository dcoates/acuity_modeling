"""Shared-eps model: one eps for both acuity groups, separate gamma and sigma per group.

Compares with the separate high-group fit (A) and the hierarchical shared-eps-and-gamma model (D)
using 5-fold CV on the high group, same folds as cv_split_hierarchical.py.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows

HERE = Path(__file__).parent
CUTOFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
N_FOLDS = 5
SEED = 0


def fit_shared_eps(kb, kw, lo, up, group, seed):
    """eps shared; gamma and sigma per group (group = 1 for high)."""
    grp = np.asarray(group, dtype=bool)
    with pm.Model():
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        log_g_lo = pm.Normal("log_g_low", mu=0.0, sigma=1.0)
        log_g_hi = pm.Normal("log_g_high", mu=0.0, sigma=1.0)
        g_lo = pm.Deterministic("gamma_low", pt.exp(log_g_lo))
        g_hi = pm.Deterministic("gamma_high", pt.exp(log_g_hi))
        g = pt.switch(pt.as_tensor(grp), g_hi, g_lo)
        s_lo = pm.HalfNormal("sigma_low", sigma=0.1)
        s_hi = pm.HalfNormal("sigma_high", sigma=0.3)
        sig = pt.switch(pt.as_tensor(grp), s_hi, s_lo)

        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(x):
            return 0.5 * (1.0 + pt.erf((x - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=seed, progressbar=False)
    return idata


def predict_shared_eps(idata, kb, kw, lo, up, group):
    post = idata.posterior.stack(s=("chain", "draw"))
    eps = post["eps"].values[None, :]
    grp = np.asarray(group, dtype=bool)[:, None]
    g = np.where(grp, post["gamma_high"].values[None, :], post["gamma_low"].values[None, :])
    sig = np.where(grp, post["sigma_high"].values[None, :], post["sigma_low"].values[None, :])
    kb_ = kb[:, None]
    kw_ = kw[:, None]
    ab = eps * kb_**g
    aw = eps * kw_**g
    k_ou = ((1.0 + ab) * kb_ + (1.0 + aw) * kw_) / (1.0 + ab + aw)
    mu = -np.log10(k_ou)
    p = norm.cdf((up[:, None] - mu) / sig) - norm.cdf((lo[:, None] - mu) / sig)
    lpd = np.log(np.mean(np.maximum(p, 1e-12), axis=1))
    return lpd, mu.mean(axis=1)


def main():
    k_better, k_worse, y = load_all_rows()
    lo_all, up_all = bin_edges(y)
    lb = -np.log10(k_better)
    high = lb > CUTOFF
    high_idx = np.where(high)[0]
    low_idx = np.where(~high)[0]

    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(high_idx), N_FOLDS)

    lpd_E = np.zeros(len(y)); pred_E = np.zeros(len(y))
    for f, test in enumerate(folds):
        train_high = np.setdiff1d(high_idx, test)
        train = np.concatenate([low_idx, train_high])
        print(f"--- fold {f + 1}/{N_FOLDS} ---")
        idm = fit_shared_eps(k_better[train], k_worse[train], lo_all[train], up_all[train],
                             high[train].astype(int), seed=40 + f)
        lpd, pr = predict_shared_eps(idm, k_better[test], k_worse[test], lo_all[test],
                                     up_all[test], np.ones(len(test), dtype=int))
        lpd_E[test] = lpd; pred_E[test] = pr

    yh = y[high_idx]
    r = yh - pred_E[high_idx]
    print("\nE shared eps, separate gamma (CV): "
          f"mean log pred. density = {lpd_E[high_idx].mean():.3f}, "
          f"RMSE = {np.sqrt(np.mean(r**2)):.4f}, bias = {r.mean():+.4f}")

    # Compare with the earlier CV results for A and D, if the CSV is present.
    prev = HERE / "ding_sperling_cv_high.csv"
    if prev.exists():
        cv = pd.read_csv(prev)
        for label, col in [("A separate (CV)", "lpd_A"), ("D shared eps+gamma (CV)", "lpd_D")]:
            print(f"{label}: mean log pred. density = {cv[col].mean():.3f}")

    full = fit_shared_eps(k_better, k_worse, lo_all, up_all, high.astype(int), seed=1)
    summary = az.summary(full, var_names=["eps", "gamma_low", "gamma_high", "sigma_low", "sigma_high"], round_to=4)
    print("\n=== Shared-eps fit, all rows ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    out = HERE / "ding_sperling_trace_shared_eps.nc"
    full.to_netcdf(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
