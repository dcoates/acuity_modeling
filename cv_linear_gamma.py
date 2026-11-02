"""Cutoff-free model: gamma and sigma vary smoothly with better-eye logMAR.

    log gamma_i = alpha + beta * (x_i - C)
    log sigma_i = s0 + s1 * (x_i - C)
    eps shared; x_i = better-eye logMAR; C = centering value (0.5)

Compared with the split models by 5-fold CV on the high group (same folds as
cv_split_hierarchical.py). Log-linear forms keep gamma and sigma positive.
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
CUTOFF = 0.4  # used only to define the high-group CV test set, not in the model
C = 0.5
N_FOLDS = 5
SEED = 0


def fit_linear(kb, kw, lo, up, x, seed):
    xc = x - C
    with pm.Model():
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0)
        s0 = pm.Normal("log_sigma0", mu=-3.0, sigma=1.0)
        s1 = pm.Normal("log_sigma_slope", mu=0.0, sigma=1.0)
        g = pt.exp(alpha + beta * xc)
        sig = pt.exp(s0 + s1 * xc)

        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(v):
            return 0.5 * (1.0 + pt.erf((v - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=seed, progressbar=False)
    return idata


def predict_linear(idata, kb, kw, lo, up, x):
    post = idata.posterior.stack(s=("chain", "draw"))
    eps = post["eps"].values[None, :]
    alpha = post["alpha"].values[None, :]
    beta = post["beta"].values[None, :]
    s0 = post["log_sigma0"].values[None, :]
    s1 = post["log_sigma_slope"].values[None, :]
    xc = (x - C)[:, None]
    g = np.exp(alpha + beta * xc)
    sig = np.exp(s0 + s1 * xc)
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
    x_all = -np.log10(k_better)
    high = x_all > CUTOFF
    high_idx = np.where(high)[0]
    low_idx = np.where(~high)[0]

    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(high_idx), N_FOLDS)

    lpd_F = np.zeros(len(y)); pred_F = np.zeros(len(y))
    for f, test in enumerate(folds):
        train_high = np.setdiff1d(high_idx, test)
        train = np.concatenate([low_idx, train_high])
        print(f"--- fold {f + 1}/{N_FOLDS} ---")
        idm = fit_linear(k_better[train], k_worse[train], lo_all[train], up_all[train],
                         x_all[train], seed=60 + f)
        lpd, pr = predict_linear(idm, k_better[test], k_worse[test], lo_all[test],
                                 up_all[test], x_all[test])
        lpd_F[test] = lpd; pred_F[test] = pr

    yh = y[high_idx]
    r = yh - pred_F[high_idx]
    print("\nF linear gamma and sigma (CV): "
          f"mean log pred. density = {lpd_F[high_idx].mean():.3f}, "
          f"RMSE = {np.sqrt(np.mean(r**2)):.4f}, bias = {r.mean():+.4f}")

    prev = HERE / "ding_sperling_cv_high.csv"
    if prev.exists():
        cv = pd.read_csv(prev)
        print(f"A separate (CV): {cv['lpd_A'].mean():.3f}   D shared eps+gamma (CV): {cv['lpd_D'].mean():.3f}")
    print("E shared eps, separate gamma/sigma (CV): -3.109 (from cv_shared_eps.py)")

    full = fit_linear(k_better, k_worse, lo_all, up_all, x_all, seed=1)
    summary = az.summary(full, var_names=["eps", "alpha", "beta", "log_sigma0", "log_sigma_slope"], round_to=4)
    print("\n=== Linear model, all rows ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    out = HERE / "ding_sperling_trace_linear.nc"
    full.to_netcdf(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
