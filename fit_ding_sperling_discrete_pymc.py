"""Ding & Sperling (Model 2) binocular acuity fit with a discretized likelihood.

The acuity charts give discrete letter/line readings, so each observed binocular
logMAR value y_i is treated as the latent continuous acuity falling in its
measurement bin. The bin for a level is the interval between the midpoints to the
neighbouring observed levels. The likelihood for each row is

    P(y_i) = Phi((upper_i - mu_i) / sigma) - Phi((lower_i - mu_i) / sigma)

where mu_i is the Model 2 prediction (see fit_ding_sperling_pymc.py).
"""

from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from fit_ding_sperling_pymc import load_pairs, to_logmar

HERE = Path(__file__).parent
OUT_TRACE = HERE / "ding_sperling_trace_discrete.nc"
OUT_SUMMARY = HERE / "ding_sperling_summary_discrete.csv"


def bin_edges(y):
    """Return (lower, upper) bin edges for each observation.

    Levels are the distinct observed values. Each bin spans the midpoints to its
    neighbours; the outer bins extend by half the adjacent gap.
    """
    levels, inverse = np.unique(np.round(y, 6), return_inverse=True)
    gaps = np.diff(levels)
    mids = levels[:-1] + gaps / 2
    lo_edge = levels[0] - gaps[0] / 2
    hi_edge = levels[-1] + gaps[-1] / 2
    edges = np.concatenate([[lo_edge], mids, [hi_edge]])
    return edges[inverse], edges[inverse + 1]


def build_model(k_better, k_worse, y):
    lower, upper = bin_edges(y)
    with pm.Model() as model:
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        log_g = pm.Normal("log_g", mu=0.0, sigma=1.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        g = pm.Deterministic("gamma", pt.exp(log_g))
        sigma = pm.HalfNormal("sigma", sigma=0.1)

        a_better = eps * k_better**g
        a_worse = eps * k_worse**g
        k_ou = ((1.0 + a_better) * k_better + (1.0 + a_worse) * k_worse) / (
            1.0 + a_better + a_worse
        )
        mu = pm.Deterministic("mu_logmar", to_logmar(k_ou))

        def cdf(x):
            return 0.5 * (1.0 + pt.erf((x - mu) / (sigma * np.sqrt(2.0))))

        prob = cdf(upper) - cdf(lower)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
    return model


def main():
    k_better, k_worse, y = load_pairs()
    print(f"Rows used: {len(y)}, distinct levels: {len(np.unique(np.round(y, 6)))}")
    model = build_model(k_better, k_worse, y)
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            random_seed=1,
            progressbar=False,
        )
        # Pointwise log-likelihood for later LOO comparison with the continuous model.
        pm.compute_log_likelihood(idata, var_names=None)

    summary = az.summary(idata, var_names=["eps", "gamma", "sigma"], round_to=4)
    print(summary.to_string())
    summary.to_csv(OUT_SUMMARY)

    pred = idata.posterior["mu_logmar"].mean(("chain", "draw")).values
    print(f"Posterior-mean RMSE (logMAR): {np.sqrt(np.mean((pred - y) ** 2)):.4f}")
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}")
    print(f"Min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")

    idata.to_netcdf(OUT_TRACE)
    print(f"Saved trace -> {OUT_TRACE}")
    print(f"Saved summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
