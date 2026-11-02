"""Fit the Ding & Sperling (2006) gain-control summation model (Model 2 in
Lew et al., VSS 2026 poster) to binocular acuity with PyMC.

Model (poster, Model 2), with k_better = better-eye k, k_worse = worse-eye k:

    k_OU = [(1 + eps*k_better**g) * k_better + (1 + eps*k_worse**g) * k_worse]
           / [1 + eps*k_better**g + eps*k_worse**g]

Units: k = -10 * logMAR + 11 (poster's VA_mod transform, positive for all
logMAR <= 1.1). The prediction is converted back with logMAR = (11 - k) / 10.

Likelihood: Normal on binocular logMAR, with a residual SD sigma.
"""

import re
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

HERE = Path(__file__).parent
DATA = HERE / "binoacuitydata_valid.xlsx"
OUT_TRACE = HERE / "ding_sperling_trace.nc"
OUT_SUMMARY = HERE / "ding_sperling_summary.csv"


def to_k(logmar):
    return 11.0 - 10.0 * np.asarray(logmar, dtype=float)


def to_logmar(k):
    return (11.0 - k) / 10.0


SC_DATA = HERE / "nonstandard_stereo_sc.xlsx"
SC_EYE_COLS = ["Last VA CC - Both.1", "Last VA CC - Left.1", "Last VA CC - Right.1"]


def load_sc_df(path: Path = SC_DATA):
    """Kept rows of the _sc file: MRN, age, raw VA (second .1 block), and parsed logMAR.

    Uses the same Snellen validity rules and integer-Age rule as the main data
    (filter_valid_acuity.py). Rows that fail are dropped.
    """
    from filter_valid_acuity import parse_logmar

    df = pd.read_excel(path)
    both_col, left_col, right_col = SC_EYE_COLS
    age = df["Age"].map(parse_age).to_numpy()
    left = df[left_col].map(parse_logmar).to_numpy()
    right = df[right_col].map(parse_logmar).to_numpy()
    both = df[both_col].map(parse_logmar).to_numpy()
    ok = np.isfinite(age) & np.isfinite(left) & np.isfinite(right) & np.isfinite(both)
    print(f"sc file: kept {ok.sum()} of {len(df)} rows")
    out = pd.DataFrame({
        "MRN": df["MRN"], "age": age,
        "raw_left": df[left_col], "raw_right": df[right_col], "raw_both": df[both_col],
        "left": left, "right": right, "both": both,
    })
    return out[ok].reset_index(drop=True)


def load_sc_logmar(path: Path = SC_DATA, with_age: bool = False):
    """Return (left, right, both) logMAR from the second (.1) VA block of the _sc file.

    With with_age=True, returns (left, right, both, age) instead.
    """
    df = load_sc_df(path)
    arrays = (df["left"].to_numpy(), df["right"].to_numpy(), df["both"].to_numpy())
    return (*arrays, df["age"].to_numpy()) if with_age else arrays


NONINT_AGE = HERE / "nonint_age.xlsx"


def parse_age(value):
    """Age in years. Integers, and single-digit ages written as '8-' or '6.' (= 8, 6). NaN otherwise."""
    m = re.fullmatch(r"\s*(\d+)\s*[-.]?\s*", str(value))
    return int(m.group(1)) if m else np.nan


def load_young_df(path: Path = NONINT_AGE):
    """Kept rows with valid acuity but a dash/dot-coded age (e.g. '8-'), as a DataFrame.

    Columns: MRN, age, raw VA (Left, Right, Both as in the file), and parsed logMAR.
    """
    from filter_valid_acuity import parse_logmar

    df = pd.read_excel(path)
    age = df["Age"].map(parse_age).to_numpy()
    left_col, right_col, both_col = "Last VA CC - Left", "Last VA CC - Right", "Last VA CC - Both"
    left = df[left_col].map(parse_logmar).to_numpy()
    right = df[right_col].map(parse_logmar).to_numpy()
    both = df[both_col].map(parse_logmar).to_numpy()
    ok = np.isfinite(age) & np.isfinite(left) & np.isfinite(right) & np.isfinite(both)
    print(f"nonint_age file: kept {ok.sum()} of {len(df)} rows (dash/dot ages with valid acuity)")
    out = pd.DataFrame({
        "MRN": df["MRN"], "age": age,
        "raw_left": df[left_col], "raw_right": df[right_col], "raw_both": df[both_col],
        "left": left, "right": right, "both": both,
    })
    return out[ok].reset_index(drop=True)


def load_young_logmar(path: Path = NONINT_AGE):
    """Return (left, right, both, age) for dash/dot-coded-age rows with valid acuity."""
    df = load_young_df(path)
    return (df["left"].to_numpy(), df["right"].to_numpy(), df["both"].to_numpy(), df["age"].to_numpy())


def load_pairs(path: Path = DATA, include_sc: bool = True):
    """Return (k_better, k_worse, logmar_binocular) for rows with valid eyes.

    Combines the main valid file with the valid rows of nonstandard_stereo_sc.xlsx.
    """
    df = pd.read_excel(path)
    left = df["Left_LogMAR"].to_numpy()
    right = df["Right_LogMAR"].to_numpy()
    binocular = df["Both_LogMAR"].to_numpy()
    if include_sc:
        sl, sr, sb = load_sc_logmar()
        left = np.concatenate([left, sl])
        right = np.concatenate([right, sr])
        binocular = np.concatenate([binocular, sb])
    # k = 11 - 10*logMAR must stay positive, so logMAR must be below 1.1.
    ok = np.isfinite(left) & np.isfinite(right) & np.isfinite(binocular)
    ok &= (left < 1.1) & (right < 1.1)
    print(f"Dropped {len(left) - ok.sum()} rows with an eye logMAR >= 1.1 (k <= 0)")
    k_left, k_right = to_k(left[ok]), to_k(right[ok])
    # Better eye = larger k (lower logMAR); worse eye = smaller k.
    k_better = np.maximum(k_left, k_right)
    k_worse = np.minimum(k_left, k_right)
    return k_better, k_worse, binocular[ok]


def build_model(k_better, k_worse, y):
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
        mu_logmar = pm.Deterministic("mu_logmar", to_logmar(k_ou))

        pm.Normal("obs", mu=mu_logmar, sigma=sigma, observed=y)
    return model


def main():
    k_better, k_worse, y = load_pairs()
    print(f"Rows used: {len(y)}")
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

    summary = az.summary(idata, var_names=["eps", "gamma", "sigma"], round_to=4)
    print(summary.to_string())
    summary.to_csv(OUT_SUMMARY)

    rmse = np.sqrt(np.mean((idata.posterior["mu_logmar"].mean(("chain", "draw")).values - y) ** 2))
    print(f"Posterior-mean RMSE (logMAR): {rmse:.4f}")
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}")
    print(f"Min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")

    idata.to_netcdf(OUT_TRACE)
    print(f"Saved trace -> {OUT_TRACE}")
    print(f"Saved summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
