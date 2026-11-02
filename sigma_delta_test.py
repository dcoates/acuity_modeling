"""Sigma test: does residual SD depend on interocular difference?

Same learned-cutoff model as cv_learned_cutoff.py, with one addition:
    sigma_i = (sigma_low + (sigma_high - sigma_low) * v_i) * exp(kappa * delta_i)
delta_i = log10(k_better / k_worse) is the interocular difference in logMAR.
kappa = 0 means no dependence on interocular difference. Its posterior tells us.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows

HERE = Path(__file__).parent


def fit_sigma_delta(kb, kw, lo, up, x, delta, seed):
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
        sig = (s_lo + (s_hi - s_lo) * v) * pt.exp(kappa * delta)

        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(z):
            return 0.5 * (1.0 + pt.erf((z - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=seed, progressbar=False)
    return idata


def main():
    k_better, k_worse, y = load_all_rows()
    lo_all, up_all = bin_edges(y)
    x = -np.log10(k_better)
    delta = np.log10(k_better / k_worse)
    idata = fit_sigma_delta(k_better, k_worse, lo_all, up_all, x, delta, seed=7)

    summary = az.summary(idata, var_names=["eps", "gamma_low", "gamma_high", "sigma_low",
                                           "sigma_high", "cutoff", "tau", "kappa"], round_to=4)
    print("=== Sigma test: sigma depends on interocular difference via kappa ===")
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    kap = idata.posterior["kappa"].values.ravel()
    print(f"P(kappa > 0) = {np.mean(kap > 0):.3f};  P(kappa < 0) = {np.mean(kap < 0):.3f}")
    print(f"Interocular difference: median {np.median(delta):.3f}, 95th percentile {np.percentile(delta, 95):.3f}")
    out = HERE / "sigma_delta_trace.nc"
    idata.to_netcdf(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
