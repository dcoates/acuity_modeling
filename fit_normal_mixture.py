"""Recording-mixture model for better-eye acuity in typically sighted adults (ages 20-50).

Latent: true better-eye logMAR, mu ~ Normal(m, s).
Recording: each record is one of two forms.
  - line-only text, e.g. "20/20"   (written with probability pi, or letter count 0)
  - line plus letters, e.g. "20/20-1" (written with probability 1 - pi)
A record's text fixes the latent interval that could have produced it:
  - letter form: |mu - r| < LETTER_HW   (one letter = 0.02 logMAR)
  - line form:   |mu - r| < LINE_HW     (rounding window, an assumption)
Likelihood, with Phi the Normal CDF:
  line-only text:  pi * P(mu in line window) + (1 - pi) * P(mu in letter window)
  letter text:     (1 - pi) * P(mu in letter window)

Rows whose raw text is neither form (e.g. trailing dash) are dropped.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from fit_ding_sperling_decimal_pymc import load_all_df

HERE = Path(__file__).parent
AGE_LO, AGE_HI = 20, 50
IDIFF_MAX = 0.1       # drop rows with interocular difference above this (logMAR)
OUT_TRACE = HERE / f"normal_mixture_trace_{AGE_LO}_{AGE_HI}_idiff{IDIFF_MAX}.nc"
LETTER_HW = 0.01
LINE_HW = 0.05
CAP_LOGMAR = None     # e.g. -0.125 for 20/15 as the chart ceiling; None = no ceiling
CAP_FLOOR = -0.5      # finite lower bound for a capped record's latent window
PRIOR_MEAN = -0.086   # target from the user
PRIOR_SD_MEAN = 0.1   # loose prior on the mean; the data should decide


def load_subset():
    df = load_all_df()
    df = df[df["age"].between(AGE_LO, AGE_HI) & (df["idiff"] <= IDIFF_MAX)].copy()
    # Better eye's raw text: the eye with the lower logMAR (ties go to left).
    raw = pd.Series(np.where(df["left"] <= df["right"], df["raw_left"], df["raw_right"]),
                    index=df.index).astype(str).str.strip()
    is_line = raw.str.fullmatch(r"20/\d+").fillna(False).astype(bool).to_numpy()
    is_letter = raw.str.fullmatch(r"20/\d+[+-]\d+").fillna(False).astype(bool).to_numpy()
    keep = is_line | is_letter
    print(f"Ages {AGE_LO}-{AGE_HI}, idiff <= {IDIFF_MAX}: {len(df)} rows; kept {keep.sum()} "
          f"(line-only {is_line.sum()}, line+letters {is_letter.sum()}; "
          f"dropped {(~keep).sum()} other forms)")
    r = df["logmar_better"].to_numpy()[keep]
    return r, is_line[keep]


def build_model(r, is_line):
    with pm.Model():
        m = pm.Normal("mean_logmar", mu=PRIOR_MEAN, sigma=PRIOR_SD_MEAN)
        s = pm.HalfNormal("sd_logmar", sigma=0.2)
        pi = pm.Beta("p_line_only", alpha=1.0, beta=1.0)

        def window_prob(half_width):
            lo = r - half_width
            hi = r + half_width
            cdf = lambda z: 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))
            return pt.clip(cdf((hi - m) / s) - cdf((lo - m) / s), 1e-12, 1.0)

        p_letter = window_prob(LETTER_HW)
        p_line = window_prob(LINE_HW)
        prob = pt.switch(is_line, pi * p_line + (1.0 - pi) * p_letter, (1.0 - pi) * p_letter)
        pm.Potential("record_loglik", pt.sum(pt.log(prob)))
    return


def main():
    r, is_line = load_subset()
    print(f"Recorded better-eye logMAR: mean {r.mean():.4f}, median {np.median(r):.4f}, "
          f"SD {r.std(ddof=1):.4f}")

    with pm.Model() as model:
        m = pm.Normal("mean_logmar", mu=PRIOR_MEAN, sigma=PRIOR_SD_MEAN)
        s = pm.HalfNormal("sd_logmar", sigma=0.2)
        pi = pm.Beta("p_line_only", alpha=1.0, beta=1.0)
        cdf = lambda z: 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))

        # Ceiling: if CAP_LOGMAR is set, line-only records at that line (e.g. the top line on
        # the chart) mean "at or better than this line", so their lower window bound is open.
        capped = (np.isclose(r, CAP_LOGMAR, atol=1e-3) & is_line) if CAP_LOGMAR is not None else np.zeros(len(r), bool)

        def window_prob(half_width):
            lo = pt.switch(capped, CAP_FLOOR, r - half_width)
            hi = r + half_width
            return pt.clip(cdf((hi - m) / s) - cdf((lo - m) / s), 1e-12, 1.0)

        p_letter = window_prob(LETTER_HW)
        p_line = window_prob(LINE_HW)
        prob = pt.switch(is_line, pi * p_line + (1.0 - pi) * p_letter, (1.0 - pi) * p_letter)
        pm.Potential("record_loglik", pt.sum(pt.log(prob)))
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=1, progressbar=False)

    summary = az.summary(idata, var_names=["mean_logmar", "sd_logmar", "p_line_only"], round_to=4)
    print(summary.to_string())
    print(f"Max R-hat: {float(summary['r_hat'].max()):.4f}, "
          f"min ESS bulk: {float(summary['ess_bulk'].min()):.0f}")
    m_post = idata.posterior["mean_logmar"].values.ravel()
    s_post = idata.posterior["sd_logmar"].values.ravel()
    print(f"Target: mean {PRIOR_MEAN:+.3f}, SD 0.095. "
          f"Posterior P(|mean - target| < 0.01) = {np.mean(np.abs(m_post - PRIOR_MEAN) < 0.01):.3f}; "
          f"P(SD in [0.085, 0.105]) = {np.mean((s_post > 0.085) & (s_post < 0.105)):.3f}")
    idata.to_netcdf(OUT_TRACE)
    print(f"Saved -> {OUT_TRACE}")


if __name__ == "__main__":
    main()
