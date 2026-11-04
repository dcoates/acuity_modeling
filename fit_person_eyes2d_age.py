"""Ages 20 and over in one fit: population by age band, recording rule and Ding & Sperling shared.

The model is fit_person_eyes2d_ext.py "asym+sig+near" (each eye's true acuity on a 2D grid, three
kinds of eye difference, poor-vision component, sigmoid gamma and binocular noise on the better
eye, one recording rule with per-line categories, style per visit, no filter). Ages under 20 are
never used.

Age bands: 20-50, over 50 to 64, over 64. In each band its own population: mean, SD, share with a
large eye difference and its SD, share with one eye worse, poor-vision share, gap and SD. Shared:
the recording rule and visit style, test noise, the small eye-difference SD and the one-eye shift.

Options (combine with "+", e.g. "shared+dnoise"):
  shared  Ding & Sperling (epsilon, gamma sigmoid, binocular noise) shared by all bands.
  dsage   epsilon, gamma_low, gamma_high, sigma_bin_low and sigma_bin_high per band too
          (cutoff and tau shared).
  dnoise  binocular noise also grows with the true eye difference d:
          sd = sqrt(sigma_bin(better eye)^2 + (k_diff * d)^2), k_diff shared.
  tnoise  eye-test noise grows for poorer eyes: sigma_test + k_test * max(a - 0.1, 0), k_test shared.
  tsig    eye-test noise rises along a sigmoid from sigma_test to sigma_test_high, centred at t_mid
          with width t_tau (all shared), so it levels off for very poor eyes.
  tstep   eye-test noise steps from sigma_test to sigma_test_high above logMAR 0.15 (between 20/25
          and 20/30); the step's place is fixed, as tsig made it a step there but could not converge.
  poorshared  the poor-vision gap and SD are shared by all bands; only its share varies by band.

Usage: python fit_person_eyes2d_age.py shared+dnoise      (BINOC_CHAINS sets the number of chains)
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import erf

import fit_person_eyes2d_ext as ext
import fit_person_eyes2d_full as full
import fit_person_records as fpr
import sampling_progress
from fit_ding_sperling_decimal_pymc import load_all_df
from fit_person_eyes2d import probs

HERE = Path(__file__).parent
OPTIONS = "asym+sig+near"
GRID = full.GRID
BANDS = {"a20": (20, 50), "a51": (50, 64), "a65": (64, 200)}
POP = ["mean_logmar", "sd_logmar", "u_eye", "sigma_d2", "v_worse", "w_poor", "gap", "s_poor"]
DS = ["ds_eps", "gamma_low", "gamma_high", "sigma_bin_low", "sigma_bin_high"]
U = pm.Uniform
PRIORS = {
    "mean_logmar": (U, dict(lower=-0.25, upper=0.3)),
    "sd_logmar": (U, dict(lower=0.01, upper=0.25)),
    "sigma_d": (U, dict(lower=0.001, upper=0.04)),
    "u_eye": (U, dict(lower=0.0, upper=0.95)),
    "sigma_d2": (U, dict(lower=0.04, upper=0.6)),
    "sigma_test": (U, dict(lower=0.01, upper=0.1)),
    "ds_eps": (U, dict(lower=0.1, upper=50.0)),
    "p_line_style": (pm.Beta, dict(alpha=1.0, beta=1.0)),
    "w_poor": (U, dict(lower=0.0, upper=0.3)),
    "gap": (U, dict(lower=0.1, upper=1.0)),
    "s_poor": (U, dict(lower=0.03, upper=0.6)),
    "b1": (U, dict(lower=-0.25, upper=0.0)),
    "b2": (U, dict(lower=0.0, upper=0.15)),
    "v_worse": (U, dict(lower=0.0, upper=0.5)),
    "delta_w": (U, dict(lower=0.02, upper=0.2)),
    "s_w": (U, dict(lower=0.01, upper=0.1)),
    "gamma_low": (pm.LogNormal, dict(mu=np.log(2.0), sigma=1.0)),
    "gamma_high": (pm.LogNormal, dict(mu=np.log(0.7), sigma=1.0)),
    "sigma_bin_low": (U, dict(lower=0.01, upper=0.15)),
    "sigma_bin_high": (U, dict(lower=0.01, upper=0.4)),
    "cutoff": (U, dict(lower=0.0, upper=1.0)),
}
EXTRA = {"dnoise": {"k_diff": (U, dict(lower=0.0, upper=1.0))},
         "tnoise": {"k_test": (U, dict(lower=0.0, upper=1.0))},
         "tsig": {"sigma_test_high": (U, dict(lower=0.02, upper=0.3)),
                  "t_mid": (U, dict(lower=0.1, upper=0.6)),
                  "t_tau": (U, dict(lower=0.02, upper=0.2))},
         "tstep": {"sigma_test_high": (U, dict(lower=0.02, upper=0.3))}}


def options(option):
    return option.split("+")


def extra(option):
    return {k: v for o in options(option) if o in EXTRA for k, v in EXTRA[o].items()}


def priors(option):
    return {**PRIORS, **extra(option)}


def varying(option):
    pop = [k for k in POP if k not in ("gap", "s_poor")] if "poorshared" in options(option) else POP
    return pop + (DS if "dsage" in options(option) else [])


def load_bands():
    df = load_all_df()
    out = {}
    for band, (lo, hi) in BANDS.items():
        keep = df["age"].between(lo, hi) if band == "a20" else (df["age"] > lo) & (df["age"] <= hi)
        d = df[keep]
        t = pd.DataFrame({"L": full.categorise(d["raw_left"], d["left"]),
                          "R": full.categorise(d["raw_right"], d["right"]),
                          "B": full.categorise(d["raw_both"], d["logmar_binocular"])}).dropna().astype(int)
        out[band] = t.groupby(["L", "R", "B"]).size().rename("n").reset_index()
        print(f"band {band}: {len(d)} visits, kept {len(t)}", flush=True)
    return out


def band_params(p, band, vary):
    q = dict(p)
    q.update({k: p[f"{k}_{band}"] for k in vary})
    return q


def band_lik(p, bands, vary, phi, stack, lib, floor):
    """Likelihood of each band's triples. Eye-record probabilities are shared; binocular ones too
    unless Ding & Sperling varies by band."""
    line_lo, line_hi, let_lo, let_hi = full.edges(p, stack)
    cuts = [(line_lo, line_hi), (let_lo, let_hi)]
    a = GRID[:, None]
    E = [floor(probs(a, ext.test_sd(a, p, lib), lo[None, :], hi[None, :], phi)) for lo, hi in cuts]

    def binocular_probs(q):
        binoc, sd = ext.ding_sperling(GRID[:, None], GRID[None, :], q, lib)
        return [floor(probs(binoc[:, :, None], sd[:, :, None], lo[None, None, :], hi[None, None, :], phi))
                for lo, hi in cuts]

    shared = None if set(DS) & set(vary) else binocular_probs(p)
    out = {}
    for band, tr in bands.items():
        q = band_params(p, band, vary)
        Bp = shared if shared is not None else binocular_probs(q)
        W = ext.acuity_weights(q, lib)
        L, R, B = tr["L"].to_numpy(), tr["R"].to_numpy(), tr["B"].to_numpy()
        parts = [lib.sum(E[s][:, L] * lib.tensordot(W[:, :, None] * Bp[s], E[s], axes=[[1], [0]])[:, B, R], axis=0)
                 for s in range(2)]
        out[band] = p["p_line_style"] * parts[0] + (1 - p["p_line_style"]) * parts[1]
    return out


def fit(bands, option):
    vary = varying(option)
    phi = lambda z: 0.5 * (1.0 + pt.erf(z / np.sqrt(2.0)))
    with pm.Model() as model:
        p = {}
        for k, (dist, kw) in priors(option).items():
            if k in vary:
                for band in BANDS:
                    p[f"{k}_{band}"] = dist(f"{k}_{band}", **kw)
            else:
                p[k] = dist(k, **kw)
        p["b3"] = pm.Deterministic("b3", p["b2"] + pm.Uniform("b_gap", lower=0.02, upper=0.2))
        p["tau"] = pm.Deterministic("tau", 0.005 + pm.HalfNormal("tau_raw", sigma=0.05))
        p.update(fpr.line_windows())
        liks = band_lik(p, bands, vary, phi, pt.stack, pt, lambda x: pt.clip(x, 1e-12, 1.0))
        for band, tr in bands.items():
            n = tr["n"].to_numpy()
            pm.Potential(f"loglik_{band}", pt.sum(n * pt.log(pt.clip(liks[band], 1e-300, 1.0))))
    return sampling_progress.sample(model, full.CHAINS, draws=500, tune=700, target_accept=0.95,
                                    nuts_sampler="nutpie", random_seed=1)


def names(option):
    vary = varying(option)
    return ([k for k in ext.names(OPTIONS) if k not in vary]
            + list(extra(option))
            + [f"{k}_{band}" for k in vary for band in BANDS])


def post_means(post, option):
    p = {v: float(post[v].mean()) for v in names(option)}
    p["h_bare"] = p["h15"]
    return p


def conditional(p, tr):
    """log p(binocular record | eye records) per visit, and the in-sample saturated value."""
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    lik = lambda t: ext.triple_lik(p, t, phi, np.array, np, lambda x: np.clip(x, 1e-12, 1.0))
    pairs = tr[["L", "R"]].drop_duplicates()
    allb = pairs.merge(pd.DataFrame({"B": range(len(full.CATS))}), how="cross").assign(n=1)
    allb["lik"] = lik(allb)
    t = tr.join(allb.groupby(["L", "R"])["lik"].sum().rename("pLR"), on=["L", "R"])
    t["lik"] = lik(tr)
    emp = t.groupby(["L", "R"])["n"].transform("sum")
    n = t["n"].sum()
    return np.sum(t.n * np.log(t.lik / t.pLR)) / n, np.sum(t.n * np.log(t.n / emp)) / n


def report(idata, bands, option):
    """Summary, then per band the checks of fit_person_eyes2d_ext.py and the conditional score.
    Chains whose mean log density is more than 10 below the best chain's are dropped first."""
    logp = idata.sample_stats["logp"].mean("draw").to_numpy()
    keep = [int(c) for c, v in zip(idata.posterior.chain.values, logp) if v > logp.max() - 10]
    print(f"chain mean logp: {np.round(logp, 1).tolist()}; keeping {len(keep)} chains", flush=True)
    idata = idata.sel(chain=keep)
    summary = az.summary(idata, var_names=names(option), round_to=4)
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    p = post_means(idata.posterior, option)
    for band, tr in bands.items():
        q = band_params(p, band, varying(option))
        print(f"\n===== band {band} {BANDS[band]}: {tr['n'].sum()} visits", flush=True)
        ext.check(q, tr)
        c, sat = conditional(q, tr)
        print(f"Conditional log p(B | L, R) per visit {c:.3f} (saturated {sat:.3f})", flush=True)


def main(option):
    bands = load_bands()
    idata = fit(bands, option)
    idata.to_netcdf(HERE / f"person_eyes2d_age_{option.replace('+', '_')}.nc")
    report(idata, bands, option)


if __name__ == "__main__":
    main(sys.argv[1])
