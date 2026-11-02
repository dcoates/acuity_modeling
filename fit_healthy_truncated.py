"""Recording-mixture model on the healthy part of the population only.

Latent better-eye logMAR is Normal(m, s), truncated above at T (eyes above T are excluded,
not modelled). Records with better-eye logMAR above T - 0.05 are dropped, so every record's
window stays below T.

Recording model is the same as fit_normal_mixture.py: a line-only text has a LINE_HW window,
a letter text has a LETTER_HW window, and line-only text comes from the line mode with
probability pi.

Per-line letter shares are compared with the truncated-Normal prediction, using the same windows.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from fit_normal_mixture import load_subset

HERE = Path(__file__).parent
T_HIGH = 0.15          # truncation: latent healthy population is below this
REC_MAX = T_HIGH - 0.05  # keep records whose windows stay below T_HIGH
MODS = [-3, -2, -1, 1, 2, 3]


def load_healthy():
    r, is_line = load_subset()
    keep = r <= REC_MAX
    print(f"Healthy subset: records <= {REC_MAX:.2f}: kept {keep.sum()} of {len(r)}")
    return r[keep], is_line[keep]


def fit(r, is_line, letter_hw, line_hw=0.05, seed=1):
    with pm.Model():
        m = pm.Normal("mean_logmar", mu=-0.086, sigma=0.1)
        s = pm.HalfNormal("sd_logmar", sigma=0.2)
        pi = pm.Beta("p_line_only", alpha=1.0, beta=1.0)
        cdf = lambda z: 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))
        Z = cdf((T_HIGH - m) / s)

        def window_prob(hw):
            lo, hi = r - hw, r + hw
            return pt.clip((cdf((hi - m) / s) - cdf((lo - m) / s)) / Z, 1e-12, 1.0)

        p_letter = window_prob(letter_hw)
        p_line = window_prob(line_hw)
        prob = pt.switch(is_line, pi * p_line + (1.0 - pi) * p_letter, (1.0 - pi) * p_letter)
        pm.Potential("record_loglik", pt.sum(pt.log(prob)))
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=seed, progressbar=False)
    return idata


def predicted_shares(m, s, letter_hw, line):
    """Truncated-Normal share of each letter modifier, for records written on the given line."""
    ell = np.log10(line / 20)
    def win(v):
        return norm.cdf((v + letter_hw - m) / s) - norm.cdf((v - letter_hw - m) / s)
    p = pd.Series({n: win(ell - 0.02 * n) for n in MODS})
    return p / p.sum()


def observed_shares(line):
    from fit_normal_mixture import load_all_df
    df = load_all_df()
    df = df[df["age"].between(20, 50) & (df["idiff"] <= 0.1)]
    raw = pd.concat([df.raw_left, df.raw_right, df.raw_both]).astype(str).str.strip()
    ex = raw.str.extract(r"^20/(\d+)([+-]\d+)$").dropna()
    ex.columns = ["line", "modifier"]
    ex["line"] = ex["line"].astype(int)
    ex["n"] = ex["modifier"].astype(int)
    sub = ex[ex["line"] == line]["n"]
    return sub.value_counts(normalize=True).reindex(MODS).fillna(0), len(sub)


def main():
    r, is_line = load_healthy()
    for letter_hw in [0.01, 0.095]:
        print(f"\n===== letter window +/-{letter_hw} =====", flush=True)
        idata = fit(r, is_line, letter_hw)
        summary = az.summary(idata, var_names=["mean_logmar", "sd_logmar", "p_line_only"], round_to=4)
        print(summary.to_string())
        print(f"Max R-hat {float(summary['r_hat'].max()):.4f}, min ESS bulk {float(summary['ess_bulk'].min()):.0f}")
        m = idata.posterior["mean_logmar"].values.mean()
        s = idata.posterior["sd_logmar"].values.mean()
        neg = [n for n in MODS if n < 0]
        print("share worse (-) letter records, model vs observed:")
        for line in [15, 20, 25]:
            p = predicted_shares(m, s, letter_hw, line)
            o, n = observed_shares(line)
            print(f"  20/{line}: model {p[neg].sum():.3f}  observed {o[neg].sum():.3f}  (n={n})")
        idata.to_netcdf(HERE / f"healthy_truncated_letter{letter_hw}.nc")


if __name__ == "__main__":
    main()
