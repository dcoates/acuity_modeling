"""Sigma-Delta interaction model, cross-validated against the learned-cutoff model.

Same learned-cutoff model as cv_learned_cutoff.py, with one addition:
    v_i     = sigmoid((x_i - cutoff) / tau)            (poor-vision weight; 0 = good, 1 = poor)
    sigma_i = (sigma_low + (sigma_high - sigma_low) * v_i) * exp(kappa * delta_i * (1 - v_i))
x = better-eye logMAR, delta = log10(k_better / k_worse).

The interocular-difference effect on spread is allowed only where vision is not poor,
since the residual table showed no Delta effect in the poor-vision band.

Evaluation: 5-fold CV over all rows, same folds as cv_cutoff_full.py. The learned-cutoff
reference scores come from ding_sperling_cutoff_full_cv.csv (same folds, same rows).
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


def fit_interaction(kb, kw, lo, up, x, delta, seed):
    with pm.Model():
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        g_lo = pm.LogNormal("gamma_low", mu=np.log(2.5), sigma=1.0)
        g_hi = pm.LogNormal("gamma_high", mu=np.log(0.7), sigma=1.0)
        s_lo = pm.HalfNormal("sigma_low", sigma=0.1)
        s_hi = pm.HalfNormal("sigma_high", sigma=0.3)
        c = pm.Uniform("cutoff", lower=0.1, upper=1.0)
        tau = pm.Deterministic("tau", 0.005 + pm.HalfNormal("tau_raw", sigma=0.05))
        kappa = pm.Normal("kappa", mu=0.0, sigma=1.0)

        v = pm.math.sigmoid((x - c) / tau)
        g = g_lo + (g_hi - g_lo) * v
        sig = (s_lo + (s_hi - s_lo) * v) * pt.exp(kappa * delta * (1.0 - v))

        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(z):
            return 0.5 * (1.0 + pt.erf((z - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm.sample(draws=1000, tune=1500, chains=4, target_accept=0.95,
                          random_seed=seed, progressbar=False)
    return idata


def predict_interaction(idata, kb, kw, lo, up, x, delta):
    post = idata.posterior.stack(s=("chain", "draw"))
    eps = post["eps"].values[None, :]
    g_lo = post["gamma_low"].values[None, :]
    g_hi = post["gamma_high"].values[None, :]
    s_lo = post["sigma_low"].values[None, :]
    s_hi = post["sigma_high"].values[None, :]
    c = post["cutoff"].values[None, :]
    tau = post["tau"].values[None, :]
    kap = post["kappa"].values[None, :]
    v = expit((x[:, None] - c) / tau)
    g = g_lo + (g_hi - g_lo) * v
    sig = (s_lo + (s_hi - s_lo) * v) * np.exp(kap * delta[:, None] * (1.0 - v))
    kb_ = kb[:, None]
    kw_ = kw[:, None]
    ab = eps * kb_**g
    aw = eps * kw_**g
    k_ou = ((1.0 + ab) * kb_ + (1.0 + aw) * kw_) / (1.0 + ab + aw)
    mu = -np.log10(k_ou)
    pr = norm.cdf((up[:, None] - mu) / sig) - norm.cdf((lo[:, None] - mu) / sig)
    lpd = np.log(np.mean(np.maximum(pr, 1e-12), axis=1))
    return lpd, mu.mean(axis=1)


def main():
    k_better, k_worse, y = load_all_rows()
    lo_all, up_all = bin_edges(y)
    x_all = -np.log10(k_better)
    delta_all = np.log10(k_better / k_worse)
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), N_FOLDS)

    lpd_all = np.zeros(len(y)); pred_all = np.zeros(len(y))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        print(f"--- fold {f + 1}/{N_FOLDS} ---", flush=True)
        idm = fit_interaction(k_better[train], k_worse[train], lo_all[train], up_all[train],
                              x_all[train], delta_all[train], seed=500 + f)
        kap = idm.posterior["kappa"].values.ravel()
        print(f"   kappa {kap.mean():.3f} [{np.percentile(kap, 3):.3f}, {np.percentile(kap, 97):.3f}]", flush=True)
        lpd, pr = predict_interaction(idm, k_better[test], k_worse[test], lo_all[test],
                                      up_all[test], x_all[test], delta_all[test])
        lpd_all[test] = lpd; pred_all[test] = pr

    row = {"model": "learned + delta x (1-v) sigma"}
    for label, mask in [("all", np.ones(len(y), bool)), ("x>0.4", x_all > 0.4), ("x>0.25", x_all > 0.25)]:
        r = y[mask] - pred_all[mask]
        row[f"lpd_{label}"] = lpd_all[mask].mean()
        row[f"bias_{label}"] = r.mean()

    ref = pd.read_csv(HERE / "ding_sperling_cutoff_full_cv.csv")
    ref = ref[ref.model == "learned"].iloc[0]
    print("\nModel                          all        x>0.4      x>0.25")
    print(f"Learned (reference)          {ref['lpd_all']:8.3f}  {ref['lpd_x>0.4']:8.3f}  {ref['lpd_x>0.25']:8.3f}")
    print(f"{row['model']:<28} {row['lpd_all']:8.3f}  {row['lpd_x>0.4']:8.3f}  {row['lpd_x>0.25']:8.3f}")
    print(f"Difference (new - learned):  {row['lpd_all'] - ref['lpd_all']:+.3f}   "
          f"{row['lpd_x>0.4'] - ref['lpd_x>0.4']:+.3f}   {row['lpd_x>0.25'] - ref['lpd_x>0.25']:+.3f}")
    pd.DataFrame([row]).to_csv(HERE / "sigma_interaction_cv.csv", index=False)

    full = fit_interaction(k_better, k_worse, lo_all, up_all, x_all, delta_all, seed=1)
    summary = az.summary(full, var_names=["eps", "gamma_low", "gamma_high", "sigma_low",
                                          "sigma_high", "cutoff", "tau", "kappa"], round_to=4)
    print("\n=== Full fit, all rows ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    full.to_netcdf(HERE / "sigma_interaction_trace.nc")
    print("Saved -> sigma_interaction_trace.nc")


if __name__ == "__main__":
    main()
