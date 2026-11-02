"""Cross-validate and fit a hierarchical version of the split Model 2 fit.

Groups split on better-eye logMAR at CUTOFF (default 0.4).

Models
  A  separate:      high-group rows only (own eps, gamma, sigma); fit per CV fold
  C  pooled-extrap: low-group rows only, applied to the high group (no high data)
  D  hierarchical:  low + high training rows; shared eps and gamma; sigma_low, sigma_high

All use decimal input (k = 10**-logMAR) and the discretized likelihood on logMAR.
Bin edges come from the full dataset's observed levels, so each held-out row has a bin.

Held-out metrics on the high group (65 rows, 5-fold CV):
  mean log predictive density (higher is better), RMSE of the posterior-mean prediction, bias.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows

HERE = Path(__file__).parent
CUTOFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
N_FOLDS = 5
SEED = 0


def make_model(kb, kw, lo, up, group, hier):
    with pm.Model():
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        log_g = pm.Normal("log_g", mu=0.0, sigma=1.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        g = pm.Deterministic("gamma", pt.exp(log_g))
        if hier:
            s_lo = pm.HalfNormal("sigma_low", sigma=0.1)
            s_hi = pm.HalfNormal("sigma_high", sigma=0.3)
            sig = pt.switch(pt.as_tensor(np.asarray(group, dtype=bool)), s_hi, s_lo)
        else:
            sig = pm.HalfNormal("sigma", sigma=0.1)

        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(x):
            return 0.5 * (1.0 + pt.erf((x - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
    return None


def fit(kb, kw, lo, up, group, hier, seed):
    import pymc as pm2
    model = pm2.Model()
    with model:
        log_eps = pm2.Normal("log_eps", mu=0.0, sigma=2.0)
        log_g = pm2.Normal("log_g", mu=0.0, sigma=1.0)
        eps = pm2.Deterministic("eps", pt.exp(log_eps))
        g = pm2.Deterministic("gamma", pt.exp(log_g))
        if hier:
            s_lo = pm2.HalfNormal("sigma_low", sigma=0.1)
            s_hi = pm2.HalfNormal("sigma_high", sigma=0.3)
            sig = pt.switch(pt.as_tensor(np.asarray(group, dtype=bool)), s_hi, s_lo)
        else:
            sig = pm2.HalfNormal("sigma", sigma=0.1)
        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm2.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(x):
            return 0.5 * (1.0 + pt.erf((x - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm2.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm2.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                           random_seed=seed, progressbar=False)
    return idata


def predict(idata, kb, kw, lo, up, group, hier):
    """Posterior predictive log density of each row's bin, and posterior-mean logMAR."""
    post = idata.posterior.stack(s=("chain", "draw"))
    eps = post["eps"].values[None, :]
    g = post["gamma"].values[None, :]
    n = len(kb)
    if hier:
        s_lo = post["sigma_low"].values[None, :]
        s_hi = post["sigma_high"].values[None, :]
        sig = np.where(np.asarray(group, dtype=bool)[:, None], s_hi, s_lo)
    else:
        sig = np.broadcast_to(post["sigma"].values[None, :], (n, eps.shape[1]))
    kb_ = kb[:, None]
    kw_ = kw[:, None]
    ab = eps * kb_**g
    aw = eps * kw_**g
    k_ou = ((1.0 + ab) * kb_ + (1.0 + aw) * kw_) / (1.0 + ab + aw)
    mu = -np.log10(k_ou)
    p = norm.cdf((up[:, None] - mu) / sig) - norm.cdf((lo[:, None] - mu) / sig)
    lpd = np.log(np.mean(np.maximum(p, 1e-12), axis=1))
    return lpd, mu.mean(axis=1)


def report(name, lpd, pred, y):
    resid = y - pred
    print(f"{name:<28} mean log pred. density = {lpd.mean():8.3f}   "
          f"RMSE = {np.sqrt(np.mean(resid**2)):.4f}   bias = {resid.mean():+.4f}")


def main():
    k_better, k_worse, y = load_all_rows()
    lo_all, up_all = bin_edges(y)
    lb = -np.log10(k_better)
    high = lb > CUTOFF
    high_idx = np.where(high)[0]
    low_idx = np.where(~high)[0]
    print(f"Low: {len(low_idx)}  High: {len(high_idx)}  (cutoff {CUTOFF})")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(high_idx)
    folds = np.array_split(perm, N_FOLDS)

    lpd_A = np.zeros(len(y)); pred_A = np.zeros(len(y))
    lpd_D = np.zeros(len(y)); pred_D = np.zeros(len(y))

    for f, test in enumerate(folds):
        train_high = np.setdiff1d(high_idx, test)
        print(f"\n--- fold {f + 1}/{N_FOLDS}: test {len(test)}, train high {len(train_high)} ---")

        # A: separate high-group model
        ia = fit(k_better[train_high], k_worse[train_high], lo_all[train_high],
                 up_all[train_high], np.zeros(len(train_high)), hier=False, seed=10 + f)
        lpd, pr = predict(ia, k_better[test], k_worse[test], lo_all[test], up_all[test],
                          np.zeros(len(test)), hier=False)
        lpd_A[test] = lpd; pred_A[test] = pr

        # D: hierarchical, low + train-high
        train = np.concatenate([low_idx, train_high])
        grp = high[train].astype(int)
        idh = fit(k_better[train], k_worse[train], lo_all[train], up_all[train], grp,
                  hier=True, seed=20 + f)
        lpd, pr = predict(idh, k_better[test], k_worse[test], lo_all[test], up_all[test],
                          np.ones(len(test), dtype=int), hier=True)
        lpd_D[test] = lpd; pred_D[test] = pr

    # C: low-group model applied to high group (no high data used)
    ic = fit(k_better[low_idx], k_worse[low_idx], lo_all[low_idx], up_all[low_idx],
             np.zeros(len(low_idx)), hier=False, seed=30)
    lpd_C, pred_C = predict(ic, k_better[high_idx], k_worse[high_idx], lo_all[high_idx],
                            up_all[high_idx], np.zeros(len(high_idx)), hier=False)

    yh = y[high_idx]
    print("\n=== Held-out performance on the high group ===")
    report("A separate (5-fold CV)", lpd_A[high_idx], pred_A[high_idx], yh)
    report("C low-model extrapolated", lpd_C, pred_C, yh)
    report("D hierarchical (5-fold CV)", lpd_D[high_idx], pred_D[high_idx], yh)

    import pandas as pd
    cv = pd.DataFrame({
        "row": high_idx, "better_logmar": lb[high_idx], "observed": yh,
        "pred_A_separate_cv": pred_A[high_idx], "lpd_A": lpd_A[high_idx],
        "pred_C_low_extrap": pred_C, "lpd_C": lpd_C,
        "pred_D_hier_cv": pred_D[high_idx], "lpd_D": lpd_D[high_idx],
    })
    cv_out = HERE / "ding_sperling_cv_high.csv"
    cv.to_csv(cv_out, index=False)
    print(f"Saved held-out predictions -> {cv_out}")

    # Final hierarchical fit on all rows, for the record
    group_all = high.astype(int)
    full = fit(k_better, k_worse, lo_all, up_all, group_all, hier=True, seed=1)
    summary = az.summary(full, var_names=["eps", "gamma", "sigma_low", "sigma_high"], round_to=4)
    print("\n=== Hierarchical fit, all rows ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    out = HERE / "ding_sperling_trace_hier.nc"
    full.to_netcdf(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
