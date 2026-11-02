"""Joint recording-rule fit: better-eye and binocular records share one latent per person.

Latent per person: g ~ Normal(m, s), truncated above at T_HIGH (healthy part only).
Records used: better eye and binocular, both as readouts of g (no summation offset).
The worse eye is not in the likelihood; it is used as a held-out check of the fitted rule.

Each record is a line-only text (window LINE_HW, probability pi) or a letter text (window
hw, probability 1 - pi). The letter window hw is free, with a HalfNormal prior, so the data
choose its width.

The per-person latent is integrated on a grid, so the likelihood is a sum over grid points.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from fit_normal_mixture import load_all_df

HERE = Path(__file__).parent
AGE_LO, AGE_HI = 20, 50
IDIFF_MAX = 0.1
T_HIGH = 0.15
LINE_HW = 0.05
GRID = np.linspace(-0.2, T_HIGH, 71)   # 0.005 spacing; coarser grid keeps memory and run time down
MODS = [-3, -2, -1, 1, 2, 3]


def record_arrays(df):
    """Better-eye and binocular records for rows where both are line-only or letter form."""
    raw_b = df["raw_left"].where(df["left"] <= df["right"], df["raw_right"]).astype(str).str.strip()
    raw_bin = df["raw_both"].astype(str).str.strip()
    logmar_b = df["logmar_better"].to_numpy()
    logmar_bin = df["logmar_binocular"].to_numpy()

    def form(s):
        is_line = s.str.fullmatch(r"20/\d+").fillna(False).astype(bool).to_numpy()
        is_letter = s.str.fullmatch(r"20/\d+[+-]\d+").fillna(False).astype(bool).to_numpy()
        return is_line, is_letter

    lb, eb = form(raw_b)
    lbi, ebi = form(raw_bin)
    keep = (lb | eb) & (lbi | ebi)
    rec = np.stack([logmar_b[keep], logmar_bin[keep]], axis=1)          # shape (n, 2)
    is_line = np.stack([lb[keep], lbi[keep]], axis=1)                    # shape (n, 2)
    print(f"Rows with better and binocular records in valid form: {keep.sum()} of {len(df)}")
    return rec, is_line


def model_loglik_terms(rec, is_line, m, s, pi, hw):
    """Per-person log-likelihood, integrated over the grid, for each candidate latent g_j."""
    edge = 0.005  # soft edge on each window, so the likelihood is smooth in hw
    cdf = lambda z: 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))
    g = pt.as_tensor_variable(GRID)                                      # (G,)
    # Normal prior on the grid (the grid stops at T_HIGH, so this is the truncated prior).
    log_prior = -0.5 * ((g - m) / s) ** 2 - pt.log(s)
    log_prior = log_prior - pt.logsumexp(log_prior)

    def window(r, half_width):
        # P(latent in [r - half_width, r + half_width]) at each grid point, soft-edged
        lo, hi = r - half_width, r + half_width
        inside = cdf((g[None, :] - lo) / edge) - cdf((g[None, :] - hi) / edge)
        return pt.clip(inside, 1e-12, 1.0)

    total = pt.zeros((rec.shape[0], GRID.size))
    for k in range(rec.shape[1]):
        r = rec[:, k][:, None]                                           # (n, 1)
        line = is_line[:, k][:, None]
        p_line = window(r, LINE_HW)
        p_letter = window(r, hw)
        rec_prob = pt.switch(line, pi * p_line + (1.0 - pi) * p_letter, (1.0 - pi) * p_letter)
        total = total + pt.log(rec_prob)
    return total + log_prior[None, :]


def fit(rec, is_line, seed=1):
    with pm.Model():
        m = pm.Normal("mean_logmar", mu=-0.086, sigma=0.1)
        s = pm.HalfNormal("sd_logmar", sigma=0.2)
        pi = pm.Beta("p_line_only", alpha=1.0, beta=1.0)
        hw = pm.HalfNormal("letter_halfwidth", sigma=0.05)
        terms = model_loglik_terms(rec, is_line, m, s, pi, hw)
        pm.Potential("records", pt.sum(pt.logsumexp(terms, axis=1)))
        idata = pm.sample(draws=500, tune=500, chains=4, cores=4, target_accept=0.9,
                          random_seed=seed, progressbar=False)
    return idata


def main():
    df = load_all_df()
    df = df[df["age"].between(AGE_LO, AGE_HI) & (df["idiff"] <= IDIFF_MAX)].copy()
    healthy = df["logmar_better"] <= T_HIGH - 0.05
    df = df[healthy]
    rec, is_line = record_arrays(df)
    idata = fit(rec, is_line)
    summary = az.summary(idata, var_names=["mean_logmar", "sd_logmar", "p_line_only",
                                           "letter_halfwidth"], round_to=4)
    print(summary.to_string())
    print(f"Max R-hat {float(summary['r_hat'].max()):.4f}, min ESS bulk {float(summary['ess_bulk'].min()):.0f}")
    idata.to_netcdf(HERE / "joint_records_trace.nc")
    print("Saved -> joint_records_trace.nc")


if __name__ == "__main__":
    main()
