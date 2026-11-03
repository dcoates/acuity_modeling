"""Extensions of fit_person_eyes2d_full.py for unequal eyes and low vision.

Options (combine with "+", e.g. "asym+sig"):
  asym  one-eye-worse component: a share v of people have one eye (left or right at random)
        shifted worse by delta_w with spread s_w, the other eye as for typical people.
        This is in addition to the symmetric two-part eye differences.
  near  with asym: restrict the shift to delta_w 0.02-0.2, s_w 0.01-0.1 (the best of the three
        modes that asym+sig split into).
  sig   Ding & Sperling gamma, and binocular noise, change with the better eye's true acuity x
        through a sigmoid: w = sigmoid((x - cutoff) / tau),
        gamma = gamma_low + (gamma_high - gamma_low) * w, likewise sigma_bin (as in cv_learned_cutoff.py).

Everything else as in fit_person_eyes2d_full.py: each eye's own true acuity on a 2D grid, test
noise, one recording rule with per-line categories down to 20/400, style per visit, ages 20-50,
no interocular-difference filter.

Usage: python fit_person_eyes2d_ext.py asym+sig      (BINOC_CHAINS sets the number of chains)
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import erf, expit

import fit_person_eyes2d_full as full
import fit_person_records as fpr
import sampling_progress
from fit_person_eyes2d import probs

HERE = Path(__file__).parent
GRID = full.GRID
BASE = ["mean_logmar", "sd_logmar", "sigma_d", "u_eye", "sigma_d2", "sigma_test",
        "ds_eps", "p_line_style", "w_poor", "gap", "s_poor",
        "h15", "h20_better", "h20_worse", "h25", "b1", "b2", "b3", "c1", "c2"]
ASYM = ["v_worse", "delta_w", "s_w"]
SIG = ["gamma_low", "gamma_high", "sigma_bin_low", "sigma_bin_high", "cutoff", "tau"]
FLAT = ["ds_gamma", "sigma_bin"]


def names(options):
    return BASE + (ASYM if "asym" in options else []) + (SIG if "sig" in options else FLAT)


def bivariate(mean_x, mean_y, var_x, var_y, cov, lib):
    det = var_x * var_y - cov ** 2
    dx, dy = GRID[:, None] - mean_x, GRID[None, :] - mean_y
    q = (var_y * dx ** 2 - 2 * cov * dx * dy + var_x * dy ** 2) / det
    mass = lib.exp(-0.5 * q) / lib.sqrt(det) * full.AREA
    return mass / lib.sum(mass)


def acuity_weights(p, lib):
    d1, d2, u = p["sigma_d"] ** 2, p["sigma_d2"] ** 2, p["u_eye"]

    def symmetric(mean, spread):
        total = 0
        for wl, vl in [(1 - u, d1), (u, d2)]:
            for wr, vr in [(1 - u, d1), (u, d2)]:
                total = total + wl * wr * bivariate(mean, mean, spread + vl, spread + vr, spread, lib)
        return total

    s2 = p["sd_logmar"] ** 2
    typical = symmetric(p["mean_logmar"], s2)
    if "v_worse" in p:
        m, shift, sw = p["mean_logmar"], p["delta_w"], p["s_w"] ** 2
        one_worse = 0.5 * (bivariate(m + shift, m, s2 + sw, s2 + d1, s2, lib)
                           + bivariate(m, m + shift, s2 + d1, s2 + sw, s2, lib))
        typical = (1 - p["v_worse"]) * typical + p["v_worse"] * one_worse
    poor = symmetric(p["mean_logmar"] + p["gap"], p["s_poor"] ** 2)
    return (1 - p["w_poor"]) * typical + p["w_poor"] * poor


def gamma_sigma(better, p, lib):
    """Gamma and binocular noise at the better eye's true acuity."""
    if "cutoff" not in p:
        return p["ds_gamma"], p["sigma_bin"]
    sig = (lambda z: 1 / (1 + np.exp(-z))) if lib is np else pt.sigmoid
    w = sig((better - p["cutoff"]) / p["tau"])
    return (p["gamma_low"] + (p["gamma_high"] - p["gamma_low"]) * w,
            p["sigma_bin_low"] + (p["sigma_bin_high"] - p["sigma_bin_low"]) * w)


def test_sd(a, p, lib):
    """Eye-test noise at true acuity a: sigma_test, rising by k_test per logMAR worse than 0.1."""
    if "k_test" not in p:
        return p["sigma_test"]
    return p["sigma_test"] + p["k_test"] * lib.maximum(a - 0.1, 0.0)


def ding_sperling(a_l, a_r, p, lib):
    gamma, sd = gamma_sigma(lib.minimum(a_l, a_r), p, lib)
    if "k_diff" in p:
        # Binocular noise grows with the true eye difference (added in variance).
        sd = lib.sqrt(sd ** 2 + (p["k_diff"] * (a_l - a_r)) ** 2)
    k_l, k_r = lib.power(10.0, -a_l), lib.power(10.0, -a_r)
    w_l, w_r = p["ds_eps"] * lib.power(k_l, gamma), p["ds_eps"] * lib.power(k_r, gamma)
    return -lib.log10(((1 + w_l) * k_l + (1 + w_r) * k_r) / (1 + w_l + w_r)), sd


def triple_lik(p, tr, phi, stack, lib, floor):
    line_lo, line_hi, let_lo, let_hi = full.edges(p, stack)
    W = acuity_weights(p, lib)
    a = GRID[:, None]
    binoc, sd_bin = ding_sperling(GRID[:, None], GRID[None, :], p, lib)
    binoc = binoc[:, :, None]
    sd_bin = sd_bin[:, :, None] if "cutoff" in p else sd_bin
    L, R, B = tr["L"].to_numpy(), tr["R"].to_numpy(), tr["B"].to_numpy()
    out = []
    for lo, hi in [(line_lo, line_hi), (let_lo, let_hi)]:
        E = floor(probs(a, test_sd(a, p, lib), lo[None, :], hi[None, :], phi))
        Bp = floor(probs(binoc, sd_bin, lo[None, None, :], hi[None, None, :], phi))
        Q = lib.tensordot(W[:, :, None] * Bp, E, axes=[[1], [0]])
        out.append(lib.sum(E[:, L] * Q[:, B, R], axis=0))
    pi = p["p_line_style"]
    return pi * out[0] + (1 - pi) * out[1]


def fit(tr, options):
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
            "ds_eps": pm.Uniform("ds_eps", lower=0.1, upper=50.0),
            "p_line_style": pm.Beta("p_line_style", alpha=1.0, beta=1.0),
            "w_poor": pm.Uniform("w_poor", lower=0.0, upper=0.2),
            "gap": pm.Uniform("gap", lower=0.1, upper=1.0),
            "s_poor": pm.Uniform("s_poor", lower=0.03, upper=0.6),
            "b1": pm.Uniform("b1", lower=-0.25, upper=0.0),
            "b2": pm.Uniform("b2", lower=0.0, upper=0.15),
        }
        p["b3"] = pm.Deterministic("b3", p["b2"] + pm.Uniform("b_gap", lower=0.02, upper=0.2))
        p.update(fpr.line_windows())
        if "asym" in options:
            p["v_worse"] = pm.Uniform("v_worse", lower=0.0, upper=0.3)
            if "near" in options:
                # The best mode of asym+sig: a small one-eye shift, poor eyes left to sigma_d2.
                p["delta_w"] = pm.Uniform("delta_w", lower=0.02, upper=0.2)
                p["s_w"] = pm.Uniform("s_w", lower=0.01, upper=0.1)
            else:
                p["delta_w"] = pm.Uniform("delta_w", lower=0.05, upper=1.2)
                p["s_w"] = pm.Uniform("s_w", lower=0.02, upper=0.6)
        if "sig" in options:
            # Priors follow cv_learned_cutoff.py.
            p["gamma_low"] = pm.LogNormal("gamma_low", mu=np.log(2.0), sigma=1.0)
            p["gamma_high"] = pm.LogNormal("gamma_high", mu=np.log(0.7), sigma=1.0)
            p["sigma_bin_low"] = pm.Uniform("sigma_bin_low", lower=0.01, upper=0.15)
            p["sigma_bin_high"] = pm.Uniform("sigma_bin_high", lower=0.01, upper=0.4)
            p["cutoff"] = pm.Uniform("cutoff", lower=0.0, upper=1.0)
            p["tau"] = pm.Deterministic("tau", 0.005 + pm.HalfNormal("tau_raw", sigma=0.05))
        else:
            p["ds_gamma"] = pm.Uniform("ds_gamma", lower=0.0, upper=8.0)
            p["sigma_bin"] = pm.Uniform("sigma_bin", lower=0.01, upper=0.15)
        lik = triple_lik(p, tr, phi, pt.stack, pt, lambda x: pt.clip(x, 1e-12, 1.0))
        pm.Potential("record_loglik", pt.sum(n * pt.log(pt.clip(lik, 1e-300, 1.0))))
    return sampling_progress.sample(model, full.CHAINS, draws=800, tune=1200, target_accept=0.95,
                                    nuts_sampler="nutpie", random_seed=1)


def post_means(idata, options):
    p = {v: float(idata.posterior[v].mean()) for v in names(options)}
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
    one_worse = np.zeros(n, bool)
    if "v_worse" in p:
        one_worse = ~poor & (rng.random(n) < p["v_worse"])
        left = rng.random(n) < 0.5
        shift = rng.normal(p["delta_w"], p["s_w"], n)
        base = mu + rng.normal(0, p["sigma_d"], n)
        a_l = np.where(one_worse & left, mu + shift, np.where(one_worse, base, a_l))
        a_r = np.where(one_worse & ~left, mu + shift, np.where(one_worse, base, a_r))
    y_l = a_l + rng.normal(0, 1, n) * test_sd(a_l, p, np)
    y_r = a_r + rng.normal(0, 1, n) * test_sd(a_r, p, np)
    mean_b, sd_b = ding_sperling(a_l, a_r, p, np)
    y_b = mean_b + rng.normal(0, 1, n) * sd_b
    line = rng.random(n) < p["p_line_style"]
    line_cuts = np.r_[p["b1"], p["b2"], p["b3"], full.POOR_MID]
    line_cat = np.r_[0, 3, 6, 8 + np.arange(len(full.POOR_LINES))]
    let_cuts = full.edges(p, np.array)[3][:-1]
    rec = lambda y: np.where(line, line_cat[np.searchsorted(line_cuts, y)], np.searchsorted(let_cuts, y))
    return pd.DataFrame({"L": rec(y_l), "R": rec(y_r), "B": rec(y_b), "y_l": y_l, "y_r": y_r, "y_b": y_b,
                         "poor": poor, "one_worse": one_worse})


def report(idata, tr, options):
    summary = az.summary(idata, var_names=names(options), round_to=4)
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    check(post_means(idata, options), tr)


def check(p, tr):
    """Log-likelihood and posterior checks of the records in tr at parameter values p."""
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    lik = triple_lik(p, tr, phi, np.array, np, lambda x: np.clip(x, 1e-12, 1.0))
    print(f"Log-likelihood at posterior means: {np.sum(tr['n'].to_numpy() * np.log(lik)):.1f}")
    sim = simulate(p, 1000000, np.random.default_rng(1))
    typ = sim[~sim.poor & ~sim.one_worse]
    better = np.minimum(typ.y_l, typ.y_r)
    for name, y in [("monocular", typ.y_l), ("better-eye", better), ("binocular", typ.y_b)]:
        print(f"Typical {name} reading: mean {y.mean():+.4f}, SD {y.std():.4f}")
    print(f"Typical binocular gain: over a single eye {(typ.y_l - typ.y_b).mean():+.4f}, "
          f"over the better eye {(better - typ.y_b).mean():+.4f}")
    if "cutoff" in p:
        for x in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
            g, s = gamma_sigma(np.array(x), p, np)
            print(f"  better eye {x:+.1f}: gamma {float(g):.2f}, binocular noise {float(s):.3f}")
    obs = tr.loc[tr.index.repeat(tr["n"]), ["L", "R", "B"]].reset_index(drop=True)
    n = len(obs)
    print("record    L obs/model   B obs/model")
    for i, c in enumerate(full.CATS):
        print(f"  {c:8s} {int((obs.L == i).sum()):5d}/{(sim.L == i).mean() * n:5.0f}   "
              f"{int((obs.B == i).sum()):5d}/{(sim.B == i).mean() * n:5.0f}")
    rep = full.REP
    diff_o, diff_s = np.abs(rep[obs.L] - rep[obs.R]), np.abs(rep[sim.L] - rep[sim.R])
    for cut in [0.1, 0.2, 0.3, 0.5]:
        print(f"  eyes differ by more than {cut}: obs {int((diff_o > cut + 1e-9).sum()):4d}  "
              f"model {(diff_s > cut + 1e-9).mean() * n:6.0f}")
    better_o = np.minimum(rep[obs.L], rep[obs.R])
    better_s = np.minimum(rep[sim.L], rep[sim.R])
    gap_o, gap_s = rep[obs.B] - better_o, rep[sim.B] - better_s
    print("  binocular minus better-eye record (obs / model), by better eye and difference:")
    for name, lo, hi in [("better <= 20/25", -1, 0.1), ("better worse than 20/25", 0.1, 9)]:
        for dname, dlo, dhi in [("diff 0", -1, 1e-9), ("diff 0-0.1", 1e-9, 0.1 + 1e-9), ("diff > 0.1", 0.1 + 1e-9, 9)]:
            ko = (better_o > lo) & (better_o <= hi) & (diff_o > dlo) & (diff_o <= dhi)
            ks = (better_s > lo) & (better_s <= hi) & (diff_s > dlo) & (diff_s <= dhi)
            print(f"    {name:24s} {dname:11s} n={int(ko.sum()):4d}  mean {gap_o[ko].mean():+.3f} / {gap_s[ks].mean():+.3f}"
                  f"  binocular worse {np.mean(gap_o[ko] > 1e-9):.2f} / {np.mean(gap_s[ks] > 1e-9):.2f}")
    patterns = {
        "all three bare 20/15": lambda d: (d.L == 0) & (d.R == 0) & (d.B == 0),
        "all three bare 20/20": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 3),
        "eyes bare 20/20, B 20/20 plus": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 2),
        "one eye bare 20/15, other bare 20/20": lambda d: ((d.L == 0) & (d.R == 3)) | ((d.L == 3) & (d.R == 0)),
        "bare 20/20 + 20/20 minus (eyes)": lambda d: ((d.L == 3) & (d.R == 4)) | ((d.L == 4) & (d.R == 3)),
    }
    for name, f in patterns.items():
        print(f"  {name:38s} obs {int(f(obs).sum()):5d}  model {f(sim).mean() * n:5.0f}")
    misplaced = sum(abs(int((obs.B == i).sum()) - (sim.B == i).mean() * n) for i in range(len(full.CATS))) / (2 * n)
    print(f"Binocular records misplaced {misplaced:.3f}", flush=True)


def main(options):
    tr = full.load_triples()
    print(f"options {options}: {tr['n'].sum()} persons, {len(tr)} triples", flush=True)
    idata = fit(tr, options)
    report(idata, tr, options)
    idata.to_netcdf(HERE / f"person_eyes2d_{options.replace('+', '_')}.nc")


if __name__ == "__main__":
    main(sys.argv[1])
