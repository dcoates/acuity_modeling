"""Ding & Sperling (Model 2) binocular acuity fit on decimal acuity, with a
discretized likelihood on logMAR.

Input scale: decimal acuity k = 10 ** (-logMAR), a ratio-scale quantity with a true
zero and 20/20 = 1. No offset is needed, and every finite logMAR gives a positive k,
so no rows are dropped.

Model 2 (Ding & Sperling 2006, as written in Lew et al., VSS 2026):

    k_OU = [(1 + eps*k_better**g) * k_better + (1 + eps*k_worse**g) * k_worse]
           / [1 + eps*k_better**g + eps*k_worse**g]

Prediction on the outcome scale: logMAR_OU = -log10(k_OU).
Likelihood: observed binocular logMAR levels treated as discretized bins of a latent
Normal on logMAR with SD sigma (same bins as fit_ding_sperling_discrete_pymc.py).
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_pymc import DATA, load_sc_df, load_sc_logmar, load_young_df, load_young_logmar

HERE = Path(__file__).parent
OUT_TRACE = HERE / "ding_sperling_trace_decimal.nc"
OUT_SUMMARY = HERE / "ding_sperling_summary_decimal.csv"


def load_all_df(path: Path = DATA):
    """Combined table of all valid rows: main file, sc file, and dash/dot-age rows.

    Columns:
      MRN               first column of the source file (kept as the row key)
      source            "main", "sc" or "young" (which input file the row came from)
      age               age in years
      raw_left, raw_right, raw_both  the three VA columns as written in the source file
      left, right       eye logMAR as recorded
      logmar_binocular  binocular logMAR
      logmar_better     better-eye logMAR (lower of left/right)
      logmar_worse      worse-eye logMAR
      k_better, k_worse decimal acuity of better and worse eye, 10 ** (-logMAR)
      k_binocular       decimal acuity of the binocular eye, 10 ** (-logmar_binocular)
      advantage         logmar_better - logmar_binocular (positive = binocular better than better eye)
      idiff             interocular difference, log10(k_better / k_worse) = logmar_worse - logmar_better
    """
    df = pd.read_excel(path)
    main = pd.DataFrame({
        "MRN": df["MRN"], "source": "main", "age": df["Age"].astype(float),
        "raw_left": df["Last_VA_CC_-_Left"], "raw_right": df["Last_VA_CC_-_Right"],
        "raw_both": df["Last_VA_CC_-_Both"],
        "left": df["Left_LogMAR"], "right": df["Right_LogMAR"], "binocular": df["Both_LogMAR"],
    })
    sc = load_sc_df().assign(source="sc").rename(columns={"both": "binocular"})
    young = load_young_df().assign(source="young").rename(columns={"both": "binocular"})
    out = pd.concat([main, sc, young], ignore_index=True)
    out = out[["MRN", "source", "age", "raw_left", "raw_right", "raw_both",
               "left", "right", "binocular"]]

    cols = ["left", "right", "binocular"]
    ok = np.isfinite(out[cols].to_numpy(dtype=float)).all(axis=1)
    print(f"Rows used: {ok.sum()} of {len(out)} (no logMAR cutoff needed)")
    out = out[ok].reset_index(drop=True).rename(columns={"binocular": "logmar_binocular"})

    out["logmar_better"] = np.minimum(out["left"], out["right"])
    out["logmar_worse"] = np.maximum(out["left"], out["right"])
    out["k_better"] = 10.0 ** (-out["logmar_better"])
    out["k_worse"] = 10.0 ** (-out["logmar_worse"])
    out["k_binocular"] = 10.0 ** (-out["logmar_binocular"])
    out["advantage"] = out["logmar_better"] - out["logmar_binocular"]
    out["idiff"] = np.log10(out["k_better"] / out["k_worse"])
    return out


def load_all_rows(path: Path = DATA):
    """Return (k_better, k_worse, logmar_binocular) for all valid rows, decimal k."""
    df = load_all_df(path)
    return df["k_better"].to_numpy(), df["k_worse"].to_numpy(), df["logmar_binocular"].to_numpy()


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
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(x):
            return 0.5 * (1.0 + pt.erf((x - mu) / (sigma * np.sqrt(2.0))))

        prob = cdf(upper) - cdf(lower)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
    return model


def main():
    k_better, k_worse, y = load_all_rows()
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
        pm.compute_log_likelihood(idata, var_names=None)

    summary = az.summary(idata, var_names=["eps", "gamma", "sigma"], round_to=4)
    print(summary.to_string())
    summary.to_csv(OUT_SUMMARY)

    pred = idata.posterior["mu_logmar"].mean(("chain", "draw")).values
    resid = y - pred
    print(f"RMSE = {np.sqrt(np.mean(resid**2)):.4f}, bias = {resid.mean():+.4f}")
    print(f"Share within +/-0.05 logMAR: {np.mean(np.abs(resid) <= 0.05):.3f}")
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}")
    print(f"Min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    hi = pred > 0.5
    print(f"Rows with pred > 0.5: {hi.sum()}, mean residual there: {resid[hi].mean():+.4f}")

    idata.to_netcdf(OUT_TRACE)
    print(f"Saved trace -> {OUT_TRACE}")


if __name__ == "__main__":
    main()
