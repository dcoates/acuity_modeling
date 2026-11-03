"""Per-person record model with each eye's own true acuity, and Ding & Sperling for binocular.

True acuity: a_L = mu + d_L, a_R = mu + d_R, with person value mu ~ Normal(m, s) and eye
differences d in two parts (share 1 - u with SD sigma_d, share u with SD sigma_d2). A small
share w_poor of people come from a poor-vision component. (a_L, a_R) is integrated on a 2D grid.
Readings: eye reading = a + test noise (SD sigma_test); binocular reading =
DingSperling(a_L, a_R) + noise (SD sigma_bin). Noise is integrated exactly within each record's
window, so the grid only has to resolve the acuity distribution.
Recording: one rule for all three readings (per-line bare windows, letter splits), style per visit.

Usage: python fit_person_eyes2d.py [idiff_max]   (default 9 = no filter)
Set BINOC_CHAINS for the number of chains.
"""

import os
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import erf

import fit_person_records as fpr
import sampling_progress

HERE = Path(__file__).parent
STEP = 0.01
GRID = np.arange(-0.4, 0.8 + STEP / 2, STEP)
CHAINS = int(os.environ.get("BINOC_CHAINS", 10))
NAMES = ["mean_logmar", "sd_logmar", "sigma_d", "u_eye", "sigma_d2", "sigma_test", "sigma_bin",
         "ds_eps", "ds_gamma", "p_line_style", "w_poor", "gap", "s_poor",
         "h15", "h20_better", "h20_worse", "h25", "b1", "b2", "b3", "c1", "c2"]


def ding_sperling(a_l, a_r, eps, gamma, lib):
    k_l, k_r = lib.power(10.0, -a_l), lib.power(10.0, -a_r)
    w_l, w_r = eps * lib.power(k_l, gamma), eps * lib.power(k_r, gamma)
    return -lib.log10(((1 + w_l) * k_l + (1 + w_r) * k_r) / (1 + w_l + w_r))


def bivariate(x, y, mean, var_x, var_y, cov, lib):
    """Bivariate Normal density on the grid, normalised to sum to 1."""
    det = var_x * var_y - cov ** 2
    dx, dy = x - mean, y - mean
    q = (var_y * dx ** 2 - 2 * cov * dx * dy + var_x * dy ** 2) / det
    dens = lib.exp(-0.5 * q) / lib.sqrt(det)
    return dens / lib.sum(dens)


def acuity_weights(p, lib):
    """Probability of each (a_L, a_R) grid cell."""
    x, y = GRID[:, None], GRID[None, :]
    s2, d1, d2, u = p["sd_logmar"] ** 2, p["sigma_d"] ** 2, p["sigma_d2"] ** 2, p["u_eye"]
    typical = 0
    for wl, vl in [(1 - u, d1), (u, d2)]:
        for wr, vr in [(1 - u, d1), (u, d2)]:
            typical = typical + wl * wr * bivariate(x, y, p["mean_logmar"], s2 + vl, s2 + vr, s2, lib)
    sp2 = p["s_poor"] ** 2
    poor = bivariate(x, y, p["mean_logmar"] + p["gap"], sp2 + d1, sp2 + d1, sp2, lib)
    return (1 - p["w_poor"]) * typical + p["w_poor"] * poor


def probs(mean, sd, lo, hi, phi):
    return phi((hi - mean) / sd) - phi((lo - mean) / sd)


def triple_lik(p, tr, phi, stack, lib, floor):
    line_lo, line_hi, let_lo, let_hi = fpr.edges(p, stack)
    W = acuity_weights(p, lib)
    a = GRID[:, None]
    binoc = ding_sperling(GRID[:, None], GRID[None, :], p["ds_eps"], p["ds_gamma"], lib)[:, :, None]
    L, R, B = tr["L"].to_numpy(), tr["R"].to_numpy(), tr["B"].to_numpy()
    out = []
    for lo, hi in [(line_lo, line_hi), (let_lo, let_hi)]:
        E = floor(probs(a, p["sigma_test"], lo[None, :], hi[None, :], phi))              # G x K
        Bp = floor(probs(binoc, p["sigma_bin"], lo[None, None, :], hi[None, None, :], phi))  # G x G x K
        M = W[:, :, None] * Bp                                                            # G x G x K
        Q = lib.tensordot(M, E, axes=[[1], [0]])                                          # G x K(B) x K(R)
        out.append(lib.sum(E[:, L] * Q[:, B, R], axis=0))                                 # per triple
    pi = p["p_line_style"]
    return pi * out[0] + (1 - pi) * out[1]


def fit(tr):
    n = tr["n"].to_numpy()
    phi = lambda z: 0.5 * (1.0 + pt.erf(z / np.sqrt(2.0)))
    with pm.Model() as model:
        p = {
            "mean_logmar": pm.Uniform("mean_logmar", lower=-0.25, upper=0.05),
            "sd_logmar": pm.Uniform("sd_logmar", lower=0.01, upper=0.2),
            "sigma_d": pm.Uniform("sigma_d", lower=0.001, upper=0.04),
            "u_eye": pm.Uniform("u_eye", lower=0.0, upper=0.95),
            "sigma_d2": pm.Uniform("sigma_d2", lower=0.04, upper=0.4),
            "sigma_test": pm.Uniform("sigma_test", lower=0.01, upper=0.1),
            "sigma_bin": pm.Uniform("sigma_bin", lower=0.01, upper=0.15),
            "ds_eps": pm.Uniform("ds_eps", lower=0.1, upper=50.0),
            "ds_gamma": pm.Uniform("ds_gamma", lower=0.0, upper=8.0),
            "p_line_style": pm.Beta("p_line_style", alpha=1.0, beta=1.0),
            "w_poor": pm.Uniform("w_poor", lower=0.0, upper=0.05),
            "gap": pm.Uniform("gap", lower=0.1, upper=0.7),
            "s_poor": pm.Uniform("s_poor", lower=0.03, upper=0.5),
            "b1": pm.Uniform("b1", lower=-0.25, upper=0.0),
            "b2": pm.Uniform("b2", lower=0.0, upper=0.15),
        }
        p["b3"] = pm.Deterministic("b3", p["b2"] + pm.Uniform("b_gap", lower=0.02, upper=0.2))
        p.update(fpr.line_windows())
        lik = triple_lik(p, tr, phi, pt.stack, pt, lambda x: pt.clip(x, 1e-12, 1.0))
        pm.Potential("record_loglik", pt.sum(n * pt.log(pt.clip(lik, 1e-300, 1.0))))
    return sampling_progress.sample(model, CHAINS, draws=800, tune=1200, target_accept=0.95,
                                    nuts_sampler="nutpie", random_seed=1)


def post_means(idata):
    p = {v: float(idata.posterior[v].mean()) for v in NAMES}
    p["h_bare"] = p["h15"]
    return p


def simulate(p, n, rng):
    poor = rng.random(n) < p["w_poor"]
    mu = np.where(poor, rng.normal(p["mean_logmar"] + p["gap"], p["s_poor"], n),
                  rng.normal(p["mean_logmar"], p["sd_logmar"], n))

    def deviation():
        sd = np.where(rng.random(n) < p["u_eye"], p["sigma_d2"], p["sigma_d"])
        return rng.normal(0, np.where(poor, p["sigma_d"], sd))

    a_l, a_r = mu + deviation(), mu + deviation()
    y_l, y_r = a_l + rng.normal(0, p["sigma_test"], n), a_r + rng.normal(0, p["sigma_test"], n)
    y_b = ding_sperling(a_l, a_r, p["ds_eps"], p["ds_gamma"], np) + rng.normal(0, p["sigma_bin"], n)
    line = rng.random(n) < p["p_line_style"]
    cuts = np.array([p["b1"], p["b2"], p["b3"]])
    let_cuts = fpr.edges(p, np.array)[3][:-1]
    rec = lambda y: np.where(line, np.array([0, 3, 6, 8])[np.searchsorted(cuts, y)], np.searchsorted(let_cuts, y))
    return pd.DataFrame({"L": rec(y_l), "R": rec(y_r), "B": rec(y_b), "y_l": y_l, "y_r": y_r, "y_b": y_b,
                         "poor": poor})


def report(idata, tr):
    post = idata.posterior
    summary = az.summary(idata, var_names=NAMES, round_to=4)
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    p = post_means(idata)
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    lik = triple_lik(p, tr, phi, np.array, np, lambda x: np.clip(x, 1e-12, 1.0))
    print(f"Log-likelihood at posterior means: {np.sum(tr['n'].to_numpy() * np.log(lik)):.1f}")
    rng = np.random.default_rng(1)
    sim = simulate(p, 1000000, rng)
    typ = sim[~sim.poor]
    better = np.minimum(typ.y_l, typ.y_r)
    for name, y in [("monocular", typ.y_l), ("better-eye", better), ("binocular", typ.y_b)]:
        print(f"Typical {name} reading: mean {y.mean():+.4f}, SD {y.std():.4f}")
    gain = better - typ.y_b
    print(f"Typical binocular gain over better eye: mean {gain.mean():+.4f}")
    obs = tr.loc[tr.index.repeat(tr["n"]), ["L", "R", "B"]].reset_index(drop=True)
    n = len(obs)
    print("record   L obs/model   B obs/model")
    for i, c in enumerate(fpr.CATS):
        print(f"  {c:8s} {int((obs.L == i).sum()):5d}/{(sim.L == i).mean() * n:5.0f}   "
              f"{int((obs.B == i).sum()):5d}/{(sim.B == i).mean() * n:5.0f}")
    patterns = {
        "all three bare 20/15": lambda d: (d.L == 0) & (d.R == 0) & (d.B == 0),
        "all three bare 20/20": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 3),
        "eyes bare 20/20, B bare 20/15": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 0),
        "eyes bare 20/15, B bare 20/20": lambda d: (d.L == 0) & (d.R == 0) & (d.B == 3),
        "eyes bare 20/20, B 20/20 plus": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 2),
        "one eye bare 20/15, other bare 20/20": lambda d: ((d.L == 0) & (d.R == 3)) | ((d.L == 3) & (d.R == 0)),
        "bare 20/20 + 20/20 minus (eyes)": lambda d: ((d.L == 3) & (d.R == 4)) | ((d.L == 4) & (d.R == 3)),
    }
    for name, f in patterns.items():
        print(f"  {name:38s} obs {int(f(obs).sum()):5d}  model {f(sim).mean() * n:5.0f}")
    k = range(len(fpr.CATS))
    misplaced = sum(abs(int((obs.B == i).sum()) - (sim.B == i).mean() * n) for i in k) / (2 * n)
    print(f"Binocular records misplaced {misplaced:.3f}", flush=True)


def main(idiff_max):
    tr = fpr.load_triples(idiff_max)
    print(f"{tr['n'].sum()} persons, {len(tr)} distinct record triples, grid {len(GRID)}^2", flush=True)
    idata = fit(tr)
    report(idata, tr)
    tag = "nofilter" if idiff_max >= 9 else f"idiff{idiff_max}"
    idata.to_netcdf(HERE / f"person_eyes2d_{tag}.nc")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 9.0)
