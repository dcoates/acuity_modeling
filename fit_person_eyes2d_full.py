"""The 2D eye model with poor vision and large interocular differences represented properly.

Same model as fit_person_eyes2d.py, with three changes:
  - Records of 20/30 and worse get one category per chart line (20/30 to 20/400, by recorded
    logMAR, letters included) in place of one ">= 20/30" category. Below 20/25 the recording
    rule is taken as rounding to the nearest line, for both recording styles.
  - The acuity grid reaches logMAR 1.4: 0.01 steps up to 0.4, then 0.03 steps.
  - The poor-vision component has the same two-part eye differences as typical eyes.
All ages 20-50 with no interocular-difference filter.

Usage: python fit_person_eyes2d_full.py     (BINOC_CHAINS sets the number of chains)
"""

import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import erf

import fit_person_records as fpr
import sampling_progress
from fit_ding_sperling_decimal_pymc import load_all_df
from fit_person_eyes2d import NAMES, ding_sperling, probs

HERE = Path(__file__).parent
GRID = np.r_[np.arange(-0.4, 0.4, 0.01), np.arange(0.4, 1.4 + 0.015, 0.03)]
WIDTH = np.gradient(GRID)
AREA = WIDTH[:, None] * WIDTH[None, :]
CHAINS = int(os.environ.get("BINOC_CHAINS", 10))
POOR_LINES = np.array([30, 40, 50, 60, 70, 80, 100, 125, 160, 200, 250, 320, 400])
POOR_ELL = np.log10(POOR_LINES / 20)
POOR_MID = (POOR_ELL[1:] + POOR_ELL[:-1]) / 2
CATS = fpr.CATS[:8] + [f"20/{n}" for n in POOR_LINES]
BIG = fpr.BIG


def categorise(raw, logmar):
    """Record categories: the eight 20/15-20/25 categories, then one per line from 20/30."""
    cat = fpr.categorise(raw)
    worse = cat == 8
    cat[worse] = 8 + np.searchsorted(POOR_MID, logmar[worse].to_numpy())
    return cat


def load_triples():
    df = load_all_df()
    df = df[df["age"].between(fpr.AGE_LO, fpr.AGE_HI)]
    t = pd.DataFrame({"L": categorise(df["raw_left"], df["left"]),
                      "R": categorise(df["raw_right"], df["right"]),
                      "B": categorise(df["raw_both"], df["logmar_binocular"])})
    print(f"ages {fpr.AGE_LO}-{fpr.AGE_HI}: {len(t)} visits, kept {t.notna().all(axis=1).sum()}")
    t = t.dropna().astype(int)
    return t.groupby(["L", "R", "B"]).size().rename("n").reset_index()


def edges(p, stack):
    """Category edges for line-only and letter visits, extended with one bin per poor line."""
    base_line_lo, base_line_hi, base_let_lo, base_let_hi = fpr.edges(p, lambda x: x)
    los = list(POOR_MID)
    his = list(POOR_MID) + [BIG]
    line_lo = stack(base_line_lo[:8] + [p["b3"]] + los)
    line_hi = stack(base_line_hi[:8] + his)
    let_lo = stack(base_let_lo[:8] + [fpr.MID[2]] + los)
    let_hi = stack(base_let_hi[:8] + his)
    return line_lo, line_hi, let_lo, let_hi


def bivariate(mean, var_x, var_y, cov, lib):
    det = var_x * var_y - cov ** 2
    dx, dy = GRID[:, None] - mean, GRID[None, :] - mean
    q = (var_y * dx ** 2 - 2 * cov * dx * dy + var_x * dy ** 2) / det
    mass = lib.exp(-0.5 * q) / lib.sqrt(det) * AREA
    return mass / lib.sum(mass)


def acuity_weights(p, lib):
    d1, d2, u = p["sigma_d"] ** 2, p["sigma_d2"] ** 2, p["u_eye"]

    def component(mean, spread):
        total = 0
        for wl, vl in [(1 - u, d1), (u, d2)]:
            for wr, vr in [(1 - u, d1), (u, d2)]:
                total = total + wl * wr * bivariate(mean, spread + vl, spread + vr, spread, lib)
        return total

    typical = component(p["mean_logmar"], p["sd_logmar"] ** 2)
    poor = component(p["mean_logmar"] + p["gap"], p["s_poor"] ** 2)
    return (1 - p["w_poor"]) * typical + p["w_poor"] * poor


def triple_lik(p, tr, phi, stack, lib, floor):
    line_lo, line_hi, let_lo, let_hi = edges(p, stack)
    W = acuity_weights(p, lib)
    a = GRID[:, None]
    binoc = ding_sperling(GRID[:, None], GRID[None, :], p["ds_eps"], p["ds_gamma"], lib)[:, :, None]
    L, R, B = tr["L"].to_numpy(), tr["R"].to_numpy(), tr["B"].to_numpy()
    out = []
    for lo, hi in [(line_lo, line_hi), (let_lo, let_hi)]:
        E = floor(probs(a, p["sigma_test"], lo[None, :], hi[None, :], phi))
        Bp = floor(probs(binoc, p["sigma_bin"], lo[None, None, :], hi[None, None, :], phi))
        Q = lib.tensordot(W[:, :, None] * Bp, E, axes=[[1], [0]])
        out.append(lib.sum(E[:, L] * Q[:, B, R], axis=0))
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
            "sigma_d2": pm.Uniform("sigma_d2", lower=0.04, upper=0.6),
            "sigma_test": pm.Uniform("sigma_test", lower=0.01, upper=0.1),
            "sigma_bin": pm.Uniform("sigma_bin", lower=0.01, upper=0.15),
            "ds_eps": pm.Uniform("ds_eps", lower=0.1, upper=50.0),
            "ds_gamma": pm.Uniform("ds_gamma", lower=0.0, upper=8.0),
            "p_line_style": pm.Beta("p_line_style", alpha=1.0, beta=1.0),
            "w_poor": pm.Uniform("w_poor", lower=0.0, upper=0.2),
            "gap": pm.Uniform("gap", lower=0.1, upper=1.0),
            "s_poor": pm.Uniform("s_poor", lower=0.03, upper=0.6),
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
        return rng.normal(0, sd)

    a_l, a_r = mu + deviation(), mu + deviation()
    y_l, y_r = a_l + rng.normal(0, p["sigma_test"], n), a_r + rng.normal(0, p["sigma_test"], n)
    y_b = ding_sperling(a_l, a_r, p["ds_eps"], p["ds_gamma"], np) + rng.normal(0, p["sigma_bin"], n)
    line = rng.random(n) < p["p_line_style"]
    line_cuts = np.r_[p["b1"], p["b2"], p["b3"], POOR_MID]
    line_cat = np.r_[0, 3, 6, 8 + np.arange(len(POOR_LINES))]
    let_cuts = edges(p, np.array)[3][:-1]
    rec = lambda y: np.where(line, line_cat[np.searchsorted(line_cuts, y)], np.searchsorted(let_cuts, y))
    return pd.DataFrame({"L": rec(y_l), "R": rec(y_r), "B": rec(y_b), "y_l": y_l, "y_r": y_r, "y_b": y_b,
                         "poor": poor})


# Typical logMAR of each category, used only to compare interocular differences.
REP = np.r_[-0.125, -0.085, -0.05, 0.0, 0.04, 0.075, 0.097, 0.13, POOR_ELL]


def report(idata, tr):
    summary = az.summary(idata, var_names=NAMES, round_to=4)
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    p = post_means(idata)
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    lik = triple_lik(p, tr, phi, np.array, np, lambda x: np.clip(x, 1e-12, 1.0))
    print(f"Log-likelihood at posterior means: {np.sum(tr['n'].to_numpy() * np.log(lik)):.1f}")
    sim = simulate(p, 1000000, np.random.default_rng(1))
    typ = sim[~sim.poor]
    better = np.minimum(typ.y_l, typ.y_r)
    for name, y in [("monocular", typ.y_l), ("better-eye", better), ("binocular", typ.y_b)]:
        print(f"Typical {name} reading: mean {y.mean():+.4f}, SD {y.std():.4f}")
    print(f"Typical binocular gain: over a single eye {(typ.y_l - typ.y_b).mean():+.4f}, "
          f"over the better eye {(better - typ.y_b).mean():+.4f}")
    obs = tr.loc[tr.index.repeat(tr["n"]), ["L", "R", "B"]].reset_index(drop=True)
    n = len(obs)
    print("record    L obs/model   B obs/model")
    for i, c in enumerate(CATS):
        print(f"  {c:8s} {int((obs.L == i).sum()):5d}/{(sim.L == i).mean() * n:5.0f}   "
              f"{int((obs.B == i).sum()):5d}/{(sim.B == i).mean() * n:5.0f}")
    diff_obs = np.abs(REP[obs.L] - REP[obs.R])
    diff_sim = np.abs(REP[sim.L] - REP[sim.R])
    for cut in [0.1, 0.2, 0.3, 0.5]:
        print(f"  eyes differ by more than {cut}: obs {int((diff_obs > cut + 1e-9).sum()):4d}  "
              f"model {(diff_sim > cut + 1e-9).mean() * n:6.0f}")
    patterns = {
        "all three bare 20/15": lambda d: (d.L == 0) & (d.R == 0) & (d.B == 0),
        "all three bare 20/20": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 3),
        "eyes bare 20/20, B bare 20/15": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 0),
        "eyes bare 20/20, B 20/20 plus": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 2),
        "one eye bare 20/15, other bare 20/20": lambda d: ((d.L == 0) & (d.R == 3)) | ((d.L == 3) & (d.R == 0)),
        "bare 20/20 + 20/20 minus (eyes)": lambda d: ((d.L == 3) & (d.R == 4)) | ((d.L == 4) & (d.R == 3)),
    }
    for name, f in patterns.items():
        print(f"  {name:38s} obs {int(f(obs).sum()):5d}  model {f(sim).mean() * n:5.0f}")
    # Binocular against the better eye's record, for people with unequal eyes.
    for name, keep_o, keep_s in [("eyes differ by > 0.2", diff_obs > 0.2 + 1e-9, diff_sim > 0.2 + 1e-9)]:
        b_o = REP[obs.B[keep_o]] - np.minimum(REP[obs.L[keep_o]], REP[obs.R[keep_o]])
        b_s = REP[sim.B[keep_s]] - np.minimum(REP[sim.L[keep_s]], REP[sim.R[keep_s]])
        print(f"  {name}: binocular minus better-eye record, obs {b_o.mean():+.3f}, model {b_s.mean():+.3f}")
    misplaced = sum(abs(int((obs.B == i).sum()) - (sim.B == i).mean() * n) for i in range(len(CATS))) / (2 * n)
    print(f"Binocular records misplaced {misplaced:.3f}", flush=True)


def main():
    tr = load_triples()
    print(f"{tr['n'].sum()} persons, {len(tr)} distinct record triples, grid {len(GRID)}^2, "
          f"{len(CATS)} categories", flush=True)
    idata = fit(tr)
    report(idata, tr)
    idata.to_netcdf(HERE / "person_eyes2d_full.nc")


if __name__ == "__main__":
    main()
