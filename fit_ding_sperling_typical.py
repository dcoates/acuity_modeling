"""Ding & Sperling Model 2 on typical eyes (ages 20-50), with no recording model.

Uses build_model from fit_ding_sperling_decimal_pymc.py on three subsets:
  all:      every valid row for ages 20-50
  idiff0.2: interocular difference <= 0.2
  good:     both eyes logMAR <= 0.3 (20/40 or better)
Recorded logMAR values are used as they stand, so rounding and the 20/15 ceiling are not modelled.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

from fit_ding_sperling_decimal_pymc import build_model, load_all_df

HERE = Path(__file__).parent
AGE_LO, AGE_HI = 20, 50
CHAINS = 7


def subset(df, tag):
    df = df[df["age"].between(AGE_LO, AGE_HI)]
    if tag == "idiff0.2":
        return df[df["idiff"] <= 0.2]
    if tag == "good":
        return df[df["logmar_worse"] <= 0.3]
    return df


def predicted(idata, k_better, k_worse):
    post = idata.posterior
    eps, g = float(post["eps"].mean()), float(post["gamma"].mean())
    a_b, a_w = eps * k_better ** g, eps * k_worse ** g
    k_ou = ((1 + a_b) * k_better + (1 + a_w) * k_worse) / (1 + a_b + a_w)
    return -np.log10(k_ou)


def main(tag):
    df = subset(load_all_df(), tag)
    k_b, k_w, y = df["k_better"].to_numpy(), df["k_worse"].to_numpy(), df["logmar_binocular"].to_numpy()
    print(f"{tag}: {len(df)} rows", flush=True)
    with build_model(k_b, k_w, y):
        idata = pm.sample(draws=1000, tune=1000, chains=CHAINS, cores=CHAINS, target_accept=0.9,
                          nuts_sampler="nutpie", random_seed=1, progressbar=False)
    summary = az.summary(idata, var_names=["eps", "gamma", "sigma"], round_to=4)
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    pred = predicted(idata, k_b, k_w)
    err = pred - y
    print(f"RMSE {np.sqrt(np.mean(err ** 2)):.4f}, bias (pred - obs) {err.mean():+.4f}")
    print(f"Mean: better eye {df['logmar_better'].mean():+.4f}, binocular {y.mean():+.4f}, predicted {pred.mean():+.4f}")
    for lo, hi, name in [(-1, 0.05, "idiff <= 0.05"), (0.05, 0.15, "0.05-0.15"), (0.15, 9, "> 0.15")]:
        keep = (df["idiff"].to_numpy() > lo) & (df["idiff"].to_numpy() <= hi)
        if keep.sum():
            print(f"  {name:14s} n={keep.sum():5d}  RMSE {np.sqrt(np.mean(err[keep] ** 2)):.4f}  bias {err[keep].mean():+.4f}")
    idata.to_netcdf(HERE / f"ding_sperling_typical_{tag}.nc")


if __name__ == "__main__":
    main(sys.argv[1])
