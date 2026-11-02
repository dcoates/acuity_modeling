"""Learned-cutoff model: the regime switch point is a parameter.

Each row has a soft weight w_i = sigmoid((x_i - c) / tau), with x_i the better-eye logMAR.
    eps:    shared across regimes
    gamma_i = gamma_lo + (gamma_hi - gamma_lo) * w_i
    sigma_i = sigma_lo + (sigma_hi - sigma_lo) * w_i
c (cutoff) ~ Uniform(0.1, 1.0); tau (transition width) ~ HalfNormal(0.05), floored at 0.005.

The cutoff is re-learned in each of 5 CV folds (high-group test rows held out), so the
cutoff is never chosen with test data. Compared with the fixed-0.4 models using the same folds.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit
from scipy.stats import norm

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows

HERE = Path(__file__).parent
FIXED_CUTOFF = 0.4  # only defines the high-group test set for CV
N_FOLDS = 5
SEED = 0


def fit_learned(kb, kw, lo, up, x, seed):
    with pm.Model():
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        g_lo = pm.LogNormal("gamma_low", mu=np.log(2.5), sigma=1.0)
        g_hi = pm.LogNormal("gamma_high", mu=np.log(0.7), sigma=1.0)
        s_lo = pm.HalfNormal("sigma_low", sigma=0.1)
        s_hi = pm.HalfNormal("sigma_high", sigma=0.3)
        c = pm.Uniform("cutoff", lower=0.1, upper=1.0)
        tau = pm.Deterministic("tau", 0.005 + pm.HalfNormal("tau_raw", sigma=0.05))

        w = pm.math.sigmoid((x - c) / tau)
        g = g_lo + (g_hi - g_lo) * w
        sig = s_lo + (s_hi - s_lo) * w

        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(v):
            return 0.5 * (1.0 + pt.erf((v - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm.sample(draws=1000, tune=1500, chains=4, target_accept=0.95,
                          random_seed=seed, progressbar=False)
    return idata


def predict_learned(idata, kb, kw, lo, up, x):
    post = idata.posterior.stack(s=("chain", "draw"))
    eps = post["eps"].values[None, :]
    g_lo = post["gamma_low"].values[None, :]
    g_hi = post["gamma_high"].values[None, :]
    s_lo = post["sigma_low"].values[None, :]
    s_hi = post["sigma_high"].values[None, :]
    c = post["cutoff"].values[None, :]
    tau = post["tau"].values[None, :]
    w = expit((x[:, None] - c) / tau)
    g = g_lo + (g_hi - g_lo) * w
    sig = s_lo + (s_hi - s_lo) * w
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
    high = x_all > FIXED_CUTOFF
    high_idx = np.where(high)[0]
    low_idx = np.where(~high)[0]

    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(high_idx), N_FOLDS)

    lpd_G = np.zeros(len(y)); pred_G = np.zeros(len(y))
    for f, test in enumerate(folds):
        train_high = np.setdiff1d(high_idx, test)
        train = np.concatenate([low_idx, train_high])
        print(f"--- fold {f + 1}/{N_FOLDS} ---")
        idm = fit_learned(k_better[train], k_worse[train], lo_all[train], up_all[train],
                          x_all[train], seed=80 + f)
        c_post = idm.posterior["cutoff"].values.ravel()
        print(f"   learned cutoff: mean {c_post.mean():.3f}, 94% [{np.percentile(c_post, 3):.3f}, {np.percentile(c_post, 97):.3f}]")
        lpd, pr = predict_learned(idm, k_better[test], k_worse[test], lo_all[test],
                                  up_all[test], x_all[test])
        lpd_G[test] = lpd; pred_G[test] = pr

    yh = y[high_idx]
    r = yh - pred_G[high_idx]
    print("\nG learned cutoff (CV): "
          f"mean log pred. density = {lpd_G[high_idx].mean():.3f}, "
          f"RMSE = {np.sqrt(np.mean(r**2)):.4f}, bias = {r.mean():+.4f}")
    print("Reference (CV, same folds): A separate -3.118, E shared-eps -3.109, "
          "D shared eps+gamma -3.519, F linear-logMAR -3.595")

    full = fit_learned(k_better, k_worse, lo_all, up_all, x_all, seed=1)
    summary = az.summary(full, var_names=["eps", "gamma_low", "gamma_high", "sigma_low",
                                          "sigma_high", "cutoff", "tau"], round_to=4)
    print("\n=== Learned-cutoff fit, all rows ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    out = HERE / "ding_sperling_trace_learned_cutoff.nc"
    full.to_netcdf(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
