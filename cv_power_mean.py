"""Generalized-summation (power-mean) model with a sigmoid-varying exponent.

Binocular decimal acuity:
    k_OU = (k_better**p + w * k_worse**p) ** (1 / p)

p (the summation exponent) depends on better-eye logMAR x through a sigmoid:
    weight  v_i = sigmoid((x_i - cutoff) / tau)
    p_i     = p_low + (p_high - p_low) * v_i      (p = 1 linear, p = 2 quadratic, large p ~ better eye)
w is the weight on the worse eye (constant).
sigma_i = sigma_low + (sigma_high - sigma_low) * v_i   (residual SD on logMAR, as in the learned model)

Likelihood: discretized latent Normal on logMAR (same bins as the Ding & Sperling fits).
Predicted logMAR = -log10(k_OU).

Evaluation: 5-fold CV over ALL rows with the same folds as cv_cutoff_full.py
(np.random.default_rng(0).permutation), scored on all rows and on logMAR > 0.4 and > 0.25.
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
N_FOLDS = 5
SEED = 0


def fit_power(kb, kw, lo, up, x, seed):
    with pm.Model():
        p_low = pm.LogNormal("p_low", mu=np.log(2.0), sigma=0.7)
        p_high = pm.LogNormal("p_high", mu=np.log(2.0), sigma=0.7)
        w_worse = pm.HalfNormal("w_worse", sigma=1.0)
        cutoff = pm.Uniform("cutoff", lower=0.1, upper=1.0)
        tau = pm.Deterministic("tau", 0.005 + pm.HalfNormal("tau_raw", sigma=0.05))
        s_lo = pm.HalfNormal("sigma_low", sigma=0.1)
        s_hi = pm.HalfNormal("sigma_high", sigma=0.3)

        v = pm.math.sigmoid((x - cutoff) / tau)
        p = p_low + (p_high - p_low) * v
        sig = s_lo + (s_hi - s_lo) * v

        k_ou = pt.power(pt.power(kb, p) + w_worse * pt.power(kw, p), 1.0 / p)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(z):
            return 0.5 * (1.0 + pt.erf((z - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=seed, progressbar=False)
    return idata


def predict_power(idata, kb, kw, lo, up, x):
    post = idata.posterior.stack(s=("chain", "draw"))
    p_low = post["p_low"].values[None, :]
    p_high = post["p_high"].values[None, :]
    w = post["w_worse"].values[None, :]
    c = post["cutoff"].values[None, :]
    tau = post["tau"].values[None, :]
    s_lo = post["sigma_low"].values[None, :]
    s_hi = post["sigma_high"].values[None, :]
    v = expit((x[:, None] - c) / tau)
    p = p_low + (p_high - p_low) * v
    sig = s_lo + (s_hi - s_lo) * v
    kb_ = kb[:, None]
    kw_ = kw[:, None]
    k_ou = (kb_**p + w * kw_**p) ** (1.0 / p)
    mu = -np.log10(k_ou)
    pr = norm.cdf((up[:, None] - mu) / sig) - norm.cdf((lo[:, None] - mu) / sig)
    lpd = np.log(np.mean(np.maximum(pr, 1e-12), axis=1))
    return lpd, mu.mean(axis=1)


def main():
    k_better, k_worse, y = load_all_rows()
    lo_all, up_all = bin_edges(y)
    x_all = -np.log10(k_better)
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), N_FOLDS)

    lpd_all = np.zeros(len(y)); pred_all = np.zeros(len(y))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        print(f"--- fold {f + 1}/{N_FOLDS} ---", flush=True)
        idm = fit_power(k_better[train], k_worse[train], lo_all[train], up_all[train],
                        x_all[train], seed=300 + f)
        c_post = idm.posterior["cutoff"].values.ravel()
        print(f"   cutoff {c_post.mean():.3f}   p_low {idm.posterior['p_low'].values.mean():.2f}   "
              f"p_high {idm.posterior['p_high'].values.mean():.2f}", flush=True)
        lpd, pr = predict_power(idm, k_better[test], k_worse[test], lo_all[test],
                                up_all[test], x_all[test])
        lpd_all[test] = lpd; pred_all[test] = pr

    row = {"model": "power-mean, sigmoid exponent"}
    for label, mask in [("all", np.ones(len(y), bool)), ("x>0.4", x_all > 0.4), ("x>0.25", x_all > 0.25)]:
        r = y[mask] - pred_all[mask]
        row[f"lpd_{label}"] = lpd_all[mask].mean()
        row[f"rmse_{label}"] = np.sqrt(np.mean(r**2))
        row[f"bias_{label}"] = r.mean()
    print(f"\npower-mean CV: all lpd {row['lpd_all']:.3f} | x>0.4 lpd {row['lpd_x>0.4']:.3f} "
          f"bias {row['bias_x>0.4']:+.4f} | x>0.25 lpd {row['lpd_x>0.25']:.3f}", flush=True)
    print("Reference (learned Ding & Sperling, same folds): all -2.399 | x>0.4 -3.160 | x>0.25 -3.324")

    pd.DataFrame([row]).to_csv(HERE / "power_mean_cv.csv", index=False)

    full = fit_power(k_better, k_worse, lo_all, up_all, x_all, seed=1)
    summary = az.summary(full, var_names=["p_low", "p_high", "w_worse", "cutoff", "tau",
                                          "sigma_low", "sigma_high"], round_to=4)
    print("\n=== Power-mean fit, all rows ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    out = HERE / "power_mean_trace.nc"
    full.to_netcdf(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
