"""Per-person model of the left, right and binocular records, to estimate the recording process.

Latent: person acuity mu ~ (1 - w) Normal(m, s) + w Normal(m + gap, s2). The second component
holds poor eyes, so the first is the typical population.
Readings: y = mu + noise, SD sigma_eye for each eye (test noise plus real eye difference);
y = mu + delta + noise, SD sigma_bin for the binocular reading.
Recording: a record is one of nine categories (bare or letters on 20/15, 20/20, 20/25, or
20/30 and worse). Records better than 20/15 count as bare 20/15, the chart ceiling.
  line style:   bare records only, with estimated boundaries b1 < b2 < b3 between lines.
  letter style: letters fix the scale. Bare means within +/-h of a line; letter records fill
                the rest, split at the midpoint between lines.
Style is drawn once per visit (STYLE_BY_VISIT) or once per reading, P(line style) = pi.

mu is integrated out on a grid and persons are aggregated to distinct record triples, so only
the 13 population and recording parameters are sampled.
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

import sampling_progress
from fit_ding_sperling_decimal_pymc import load_all_df

HERE = Path(__file__).parent
AGE_LO, AGE_HI = 20, 50
CATS = ["15 bare", "15-", "20+", "20 bare", "20-", "25+", "25 bare", "25-", ">=30"]
L15, L20, L25, L30 = np.log10(np.array([15, 20, 25, 30]) / 20)
MID = [(L15 + L20) / 2, (L20 + L25) / 2, (L25 + L30) / 2]   # letter records split here
GRID = np.arange(-0.4, 0.8, float(os.environ.get("BINOC_GRID", 0.004)))
BIG = 10.0
CORES = int(os.environ.get("BINOC_CHAINS", 7))  # chains per run, one core each; 7 lets three runs share 24
PARAMS = ["mean_logmar", "sd_logmar", "sigma_eye", "sigma_bin", "delta_bin", "p_line_style",
          "h_bare", "b1", "b2", "b3", "w_poor", "gap", "s_poor"]
SPLITS = ["c1", "c2", "h15", "h20_better", "h20_worse", "h25",  # optional: letter splits, per-line windows
          "h15_bin", "h20_better_bin", "h20_worse_bin", "h25_bin",  # optional: binocular's own rule
          "b1_bin", "b2_bin", "b3_bin", "c1_bin", "c2_bin", "slope_bin", "xi_bin",
          "lam_bin", "bin_shift", "sigma_bin2", "w_typ2", "pop_shift", "s_typ2", "m_bin", "s_bin",
          "w_poor_bin", "gap_bin", "s_poor_bin", "u_eye", "sigma_eye2", "ds_eps", "ds_gamma"]
EYES_ONLY = False     # set by main: leave binocular records out of the likelihood
PER_LINE = ("lines", "lines_bin", "bin_ceiling", "bin_20", "bin_slope", "bin_style",
            "bin_mix", "bin_mix_sd", "pop_mix", "bin_indep", "bin_indep_mix", "bin_indep_sep",
            "bin_indep_b1", "eyes", "eyes_mix", "eyes_wide", "ds")


def categorise(s):
    ex = s.astype(str).str.strip().str.extract(r"^20/(\d+)\s*([+-]?)\s*\d*$")
    line, sign = ex[0].astype(float), ex[1].fillna("")
    out = pd.Series(np.nan, index=s.index)
    out[line <= 15] = 0
    out[(line == 15) & (sign == "-")] = 1
    for k, ln in enumerate([20, 25]):
        out[(line == ln) & (sign == "+")] = 2 + 3 * k
        out[(line == ln) & (sign == "")] = 3 + 3 * k
        out[(line == ln) & (sign == "-")] = 4 + 3 * k
    out[line >= 30] = 8
    return out


def load_triples(idiff_max):
    df = load_all_df()
    df = df[df["age"].between(AGE_LO, AGE_HI) & (df["idiff"] <= idiff_max)]
    t = pd.DataFrame({"L": categorise(df["raw_left"]), "R": categorise(df["raw_right"]),
                      "B": categorise(df["raw_both"])})
    print(f"idiff <= {idiff_max}: {len(t)} persons, kept {t.notna().all(axis=1).sum()}")
    t = t.dropna().astype(int)
    return t.groupby(["L", "R", "B"]).size().rename("n").reset_index()


def cat_probs(mean, sd, lo, hi, phi):
    """Grid by category probabilities for one reading type and one style."""
    return phi((hi - mean) / sd) - phi((lo - mean) / sd)


def edges(p, stack):
    """Lower and upper category edges for line style and letter style."""
    h, b1, b2, b3 = p["h_bare"], p["b1"], p["b2"], p["b3"]
    c1, c2 = p.get("c1", MID[0]), p.get("c2", MID[1])
    line_lo = stack([-BIG, BIG, BIG, b1, BIG, BIG, b2, BIG, b3])
    line_hi = stack([b1, BIG, BIG, b2, BIG, BIG, b3, BIG, BIG])
    h20b, h20w, h25 = p.get("h20_better", h), p.get("h20_worse", h), p.get("h25", h)
    let_lo = stack([-BIG, L15 + h, c1, -h20b, h20w, c2, L25 - h25, L25 + h25, MID[2]])
    let_hi = stack([L15 + h, c1, -h20b, h20w, c2, L25 - h25, L25 + h25, MID[2], BIG])
    return line_lo, line_hi, let_lo, let_hi


def eye_noise(p, rng, n):
    """Eye deviations from the person value, one or two parts."""
    sd = np.full(n, p["sigma_eye"])
    if "u_eye" in p:
        sd[rng.random(n) < p["u_eye"]] = p["sigma_eye2"]
    return rng.normal(0, sd)


def independent_binocular(p, rng, n):
    """Typical binocular readings drawn with no link to person acuity."""
    y = rng.normal(p["m_bin"], p["s_bin"], n)
    if "lam_bin" in p:
        alt = rng.random(n) < p["lam_bin"]
        y[alt] = rng.normal(p["m_bin"] + p["bin_shift"], p["sigma_bin2"], alt.sum())
    return y


def ds_gain(mu, p):
    """Ding & Sperling Model 2 binocular gain in logMAR when both eyes read mu."""
    lib = np if isinstance(p["ds_eps"], float) else pt
    a = p["ds_eps"] * lib.power(10.0, -p["ds_gamma"] * mu)    # eps * k ** gamma, k = 10 ** -mu
    return lib.log10(2 * (1 + a) / (1 + 2 * a))


def binocular_mean(mu, p):
    """Mean binocular reading for person acuity mu; slope 1 means a constant offset."""
    if "ds_eps" in p:
        return mu - ds_gain(mu, p)
    return p["mean_logmar"] + p["delta_bin"] + p.get("slope_bin", 1.0) * (mu - p["mean_logmar"])


def triple_lik(p, tr, by_visit, phi, stack, floor):
    """Likelihood of each record triple given grid mu, plus the population weights."""
    line_lo, line_hi, let_lo, let_hi = edges(p, stack)
    bline_lo, bline_hi, blet_lo, blet_hi = edges(p.get("bin", p), stack)
    eye = GRID[:, None]
    bin_ = binocular_mean(GRID[:, None], p)

    def eye_probs(lo, hi):
        # Most eyes differ little from the person value; a share u_eye can differ a lot.
        main = floor(cat_probs(eye, p["sigma_eye"], lo[None, :], hi[None, :], phi))
        if "u_eye" not in p:
            return main
        wide = floor(cat_probs(eye, p["sigma_eye2"], lo[None, :], hi[None, :], phi))
        return (1 - p["u_eye"]) * main + p["u_eye"] * wide

    eye_line = eye_probs(line_lo, line_hi)
    eye_let = eye_probs(let_lo, let_hi)

    def bin_probs(lo, hi):
        """Binocular record probabilities, from one offset or a mix of two."""
        if "m_bin" in p:    # independent of person acuity: every grid row is the same
            flat = 0.0 * GRID[:, None] + p["m_bin"]
            own = (1 - p["w_poor_bin"]) * cat_probs(flat, p["s_bin"], lo[None, :], hi[None, :], phi)
            own = own + p["w_poor_bin"] * cat_probs(flat + p["gap_bin"], p["s_poor_bin"],
                                                    lo[None, :], hi[None, :], phi)
            if "lam_bin" in p:
                other = cat_probs(flat + p["bin_shift"], p["sigma_bin2"], lo[None, :], hi[None, :], phi)
                own = (1 - p["lam_bin"]) * own + p["lam_bin"] * other
            return floor(own)
        main = floor(cat_probs(bin_, p["sigma_bin"], lo[None, :], hi[None, :], phi))
        if "lam_bin" not in p:
            return main
        other = floor(cat_probs(bin_ + p["bin_shift"], p.get("sigma_bin2", p["sigma_bin"]),
                                lo[None, :], hi[None, :], phi))
        return (1 - p["lam_bin"]) * main + p["lam_bin"] * other

    bin_line = bin_probs(bline_lo, bline_hi)
    bin_let = bin_probs(blet_lo, blet_hi)
    L, R, B = tr["L"].to_numpy(), tr["R"].to_numpy(), tr["B"].to_numpy()
    pi = p["p_line_style"]
    if EYES_ONLY:
        return pi * eye_line[:, L] * eye_line[:, R] + (1 - pi) * eye_let[:, L] * eye_let[:, R]
    if by_visit:
        # xi: on a letter visit, the chance the binocular reading is still recorded line-only.
        xi = p.get("xi_bin", 0.0)
        return (pi * eye_line[:, L] * eye_line[:, R] * bin_line[:, B]
                + (1 - pi) * eye_let[:, L] * eye_let[:, R] * (xi * bin_line[:, B] + (1 - xi) * bin_let[:, B]))
    return ((pi * eye_line[:, L] + (1 - pi) * eye_let[:, L])
            * (pi * eye_line[:, R] + (1 - pi) * eye_let[:, R])
            * (pi * bin_line[:, B] + (1 - pi) * bin_let[:, B]))


def pop_weights(p, exp, total):
    a = exp(-0.5 * ((GRID - p["mean_logmar"]) / p["sd_logmar"]) ** 2) / p["sd_logmar"]
    if "w_typ2" in p:
        other = exp(-0.5 * ((GRID - p["mean_logmar"] - p["pop_shift"]) / p["s_typ2"]) ** 2) / p["s_typ2"]
        a = (1 - p["w_typ2"]) * a + p["w_typ2"] * other
    m2 = p["mean_logmar"] + p["gap"]
    b = exp(-0.5 * ((GRID - m2) / p["s_poor"]) ** 2) / p["s_poor"]
    w = (1 - p["w_poor"]) * a + p["w_poor"] * b
    return w / total(w)


def line_windows(suffix=""):
    """Per-line bare windows and letter splits. Splits are tied to the windows to stay ordered."""
    w = {"h_bare": pm.Uniform(f"h15{suffix}", lower=0.005, upper=0.08),
         "h20_better": pm.Uniform(f"h20_better{suffix}", lower=0.005, upper=0.05),
         "h20_worse": pm.Uniform(f"h20_worse{suffix}", lower=0.005, upper=0.05),
         "h25": pm.Uniform(f"h25{suffix}", lower=0.001, upper=0.04)}
    low = L15 + w["h_bare"]
    w["c1"] = pm.Deterministic(f"c1{suffix}", low + pm.Beta(f"u1{suffix}", 1.0, 1.0) * (-w["h20_better"] - low))
    high = L25 - w["h25"]
    w["c2"] = pm.Deterministic(f"c2{suffix}", w["h20_worse"] + pm.Beta(f"u2{suffix}", 1.0, 1.0) * (high - w["h20_worse"]))
    return w


def fit(tr, by_visit, free_split=False):
    n = tr["n"].to_numpy()
    phi = lambda z: 0.5 * (1.0 + pt.erf(z / np.sqrt(2.0)))
    with pm.Model():
        p = {
            # Bounds keep the typical component on the grid and away from a zero-width spike.
            "mean_logmar": pm.Uniform("mean_logmar", lower=-0.25, upper=0.05),
            "sd_logmar": pm.Uniform("sd_logmar", lower=0.01, upper=0.2),
            "sigma_eye": (pm.Uniform("sigma_eye", lower=0.001, upper=0.04) if free_split in ("eyes_wide", "ds")
                          else pm.Uniform("sigma_eye", lower=0.005, upper=0.15)),
            "sigma_bin": pm.Uniform("sigma_bin", lower=0.005, upper=0.15),
            "delta_bin": pm.Normal("delta_bin", mu=0.0, sigma=0.05),
            "p_line_style": pm.Beta("p_line_style", alpha=1.0, beta=1.0),
            "b1": pm.Uniform("b1", lower=-0.25, upper=0.0),
            "b2": pm.Uniform("b2", lower=0.0, upper=0.15),
            # Capped: larger shares let the poor component swallow the typical population.
            "w_poor": (pm.Uniform("w_poor", lower=0.0, upper=0.05) if free_split in PER_LINE
                       else pm.Beta("w_poor", alpha=1.0, beta=9.0)),
            "gap": pm.Uniform("gap", lower=0.1, upper=0.7),
            "s_poor": pm.Uniform("s_poor", lower=0.03, upper=0.5),
        }
        p["b3"] = pm.Deterministic("b3", p["b2"] + pm.Uniform("b_gap", lower=0.02, upper=0.2))
        per_line = free_split in PER_LINE
        if not per_line:
            p["h_bare"] = pm.Uniform("h_bare", lower=0.005, upper=0.045)
        if per_line:
            p.update(line_windows())
            if free_split == "bin_ceiling":     # binocular differs only around 20/15
                h15_bin = pm.Uniform("h15_bin", lower=0.005, upper=0.08)
                low = L15 + h15_bin
                p["bin"] = {**p, "h_bare": h15_bin, "b1": pm.Uniform("b1_bin", lower=-0.25, upper=0.0),
                            "c1": pm.Deterministic("c1_bin", low + pm.Beta("u1_bin", 1.0, 1.0)
                                                   * (-p["h20_better"] - low))}
            elif free_split == "bin_20":        # binocular differs only in the 20/20 letter window
                p["bin"] = {**p, "h20_better": pm.Uniform("h20_better_bin", lower=0.005, upper=0.05),
                            "h20_worse": pm.Uniform("h20_worse_bin", lower=0.005, upper=0.05)}
            elif free_split == "ds":            # two-part eyes; binocular from Ding & Sperling
                p["u_eye"] = pm.Uniform("u_eye", lower=0.0, upper=0.95)
                p["sigma_eye2"] = pm.Uniform("sigma_eye2", lower=0.04, upper=0.4)
                p["ds_eps"] = pm.Uniform("ds_eps", lower=0.1, upper=50.0)
                p["ds_gamma"] = pm.Uniform("ds_gamma", lower=0.0, upper=8.0)
            elif free_split == "eyes_wide":     # two-part, with wide limits
                p["u_eye"] = pm.Uniform("u_eye", lower=0.0, upper=0.95)
                p["sigma_eye2"] = pm.Uniform("sigma_eye2", lower=0.04, upper=0.4)
            elif free_split == "eyes_mix":      # two-part eye differences
                p["u_eye"] = pm.Uniform("u_eye", lower=0.0, upper=0.3)
                p["sigma_eye2"] = pm.Uniform("sigma_eye2", lower=0.05, upper=0.4)
            elif free_split in ("bin_indep", "bin_indep_mix", "bin_indep_sep", "bin_indep_b1"):
                # Binocular has its own distribution, with no link to person acuity.
                p["m_bin"] = pm.Uniform("m_bin", lower=-0.25, upper=0.05)
                p["s_bin"] = pm.Uniform("s_bin", lower=0.01, upper=0.2)
                p["w_poor_bin"] = pm.Uniform("w_poor_bin", lower=0.0, upper=0.05)
                p["gap_bin"] = pm.Uniform("gap_bin", lower=0.1, upper=0.7)
                p["s_poor_bin"] = pm.Uniform("s_poor_bin", lower=0.03, upper=0.5)
                if free_split == "bin_indep_b1":    # one rule except the line-only 20/15 boundary
                    p["bin"] = {**p, "b1": pm.Uniform("b1_bin", lower=-0.25, upper=0.0)}
                if free_split == "bin_indep_mix":
                    p["lam_bin"] = pm.Uniform("lam_bin", lower=0.0, upper=0.5)
                    p["bin_shift"] = pm.Uniform("bin_shift", lower=-0.15, upper=0.15)
                    p["sigma_bin2"] = pm.Uniform("sigma_bin2", lower=0.005, upper=0.15)
                if free_split == "bin_indep_sep":
                    p["bin"] = {**p, **line_windows("_bin"),
                                "b1": pm.Uniform("b1_bin", lower=-0.25, upper=0.0),
                                "b2": pm.Uniform("b2_bin", lower=0.0, upper=0.15)}
                    p["bin"]["b3"] = pm.Deterministic(
                        "b3_bin", p["bin"]["b2"] + pm.Uniform("b_gap_bin", lower=0.02, upper=0.2))
            elif free_split in ("bin_mix", "bin_mix_sd"):   # shared rule; two binocular gains
                p["lam_bin"] = pm.Uniform("lam_bin", lower=0.0, upper=0.5)
                p["bin_shift"] = pm.Uniform("bin_shift", lower=-0.15, upper=0.15)
                if free_split == "bin_mix_sd":
                    p["sigma_bin2"] = pm.Uniform("sigma_bin2", lower=0.005, upper=0.15)
            elif free_split == "pop_mix":       # shared rule; typical population is two Normals
                p["w_typ2"] = pm.Uniform("w_typ2", lower=0.0, upper=0.5)
                p["pop_shift"] = pm.Uniform("pop_shift", lower=-0.15, upper=0.15)
                p["s_typ2"] = pm.Uniform("s_typ2", lower=0.01, upper=0.2)
            elif free_split == "bin_style":     # shared rule; letters are written less for binocular
                p["xi_bin"] = pm.Beta("xi_bin", alpha=1.0, beta=1.0)
            elif free_split == "bin_slope":     # shared rule; binocular gain depends on acuity
                p["slope_bin"] = pm.Uniform("slope_bin", lower=0.5, upper=1.2)
            elif free_split == "lines_bin":
                p["bin"] = {**p, **line_windows("_bin"),
                            "b1": pm.Uniform("b1_bin", lower=-0.25, upper=0.0),
                            "b2": pm.Uniform("b2_bin", lower=0.0, upper=0.15)}
                p["bin"]["b3"] = pm.Deterministic(
                    "b3_bin", p["bin"]["b2"] + pm.Uniform("b_gap_bin", lower=0.02, upper=0.2))
        elif free_split:
            p["c1"] = pm.Uniform("c1", lower=-0.1, upper=-0.03)
            p["c2"] = pm.Uniform("c2", lower=0.03, upper=0.075)
        lik = triple_lik(p, tr, by_visit, phi, pt.stack, lambda x: pt.clip(x, 1e-10, 1.0))
        weights = pop_weights(p, pt.exp, pt.sum)
        pm.Potential("record_loglik", pt.sum(n * pt.log(pt.dot(weights, lik))))
        return sampling_progress.sample(pm.modelcontext(None), CORES, draws=800, tune=1500,
                                        target_accept=0.95, nuts_sampler="nutpie", random_seed=1)


def post_means(idata):
    p = {v: float(idata.posterior[v].mean()) for v in PARAMS + SPLITS if v in idata.posterior}
    if "h15" in p:
        p["h_bare"] = p["h15"]
    names = {"h_bare": "h15_bin", "h20_better": "h20_better_bin", "h20_worse": "h20_worse_bin",
             "h25": "h25_bin", "b1": "b1_bin", "b2": "b2_bin", "b3": "b3_bin",
             "c1": "c1_bin", "c2": "c2_bin"}
    own = {k: float(idata.posterior[v].mean()) for k, v in names.items() if v in idata.posterior}
    if own:
        p["bin"] = {**p, **own}
    return p


def numpy_lik(p, tr, by_visit):
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    lik = triple_lik(p, tr, by_visit, phi, np.array, lambda x: np.clip(x, 1e-10, 1.0))
    return lik, pop_weights(p, np.exp, np.sum)


def loglik(p, tr, by_visit):
    lik, w = numpy_lik(p, tr, by_visit)
    return float(np.sum(tr["n"].to_numpy() * np.log(w @ lik)))


def bare_meaning(p, tr, by_visit):
    """Posterior of a person's mu given each record of the left eye, from all triples."""
    lik, w = numpy_lik(p, tr, by_visit)
    n = tr["n"].to_numpy()
    post = (w[:, None] * lik) / (w @ lik)    # grid by triple, sums to 1 over grid
    print("\nPerson acuity given left-eye record (mean, SD, P(better than 20/20)):")
    for k, name in enumerate(CATS):
        rows = tr["L"].to_numpy() == k
        if n[rows].sum() == 0:
            continue
        g = post[:, rows] @ n[rows] / n[rows].sum()
        mean = g @ GRID
        sd = np.sqrt(g @ (GRID - mean) ** 2)
        print(f"  {name:8s} n={n[rows].sum():5d}  mean {mean:+.3f}  SD {sd:.3f}  "
              f"P(mu<0) {g[GRID < 0].sum():.2f}")


def simulate(p, n_person, by_visit, seed=1):
    """Draw record triples from the fitted model, to compare with the data."""
    rng = np.random.default_rng(seed)
    poor = rng.random(n_person) < p["w_poor"]
    mu = np.where(poor, rng.normal(p["mean_logmar"] + p["gap"], p["s_poor"], n_person),
                  rng.normal(p["mean_logmar"], p["sd_logmar"], n_person))
    if "w_typ2" in p:
        alt = ~poor & (rng.random(n_person) < p["w_typ2"])
        mu[alt] = rng.normal(p["mean_logmar"] + p["pop_shift"], p["s_typ2"], alt.sum())
    y_bin = binocular_mean(mu, p) + rng.normal(0, p["sigma_bin"], n_person)
    if "lam_bin" in p:
        alt = rng.random(n_person) < p["lam_bin"]
        y_bin[alt] = (binocular_mean(mu[alt], p) + p["bin_shift"]
                      + rng.normal(0, p.get("sigma_bin2", p["sigma_bin"]), alt.sum()))
    if "m_bin" in p:
        y_bin = independent_binocular(p, rng, n_person)
        low_vision = rng.random(n_person) < p["w_poor_bin"]
        y_bin[low_vision] = rng.normal(p["m_bin"] + p["gap_bin"], p["s_poor_bin"], low_vision.sum())
    line_cat = np.array([0, 3, 6, 8])

    def record(y, style_line, q=p):
        line_cuts = np.array([q["b1"], q["b2"], q["b3"]])
        return np.where(style_line, line_cat[np.searchsorted(line_cuts, y)],
                        np.searchsorted(edges(q, np.array)[3][:-1], y))

    pi = p["p_line_style"]
    visit = rng.random(n_person) < pi
    styles = ([visit, visit, visit | (rng.random(n_person) < p.get("xi_bin", 0.0))] if by_visit
              else [rng.random(n_person) < pi for _ in range(3)])
    return pd.DataFrame({
        "L": record(mu + eye_noise(p, rng, n_person), styles[0]),
        "R": record(mu + eye_noise(p, rng, n_person), styles[1]),
        "B": record(y_bin, styles[2], p.get("bin", p)),
    })


def check(p, tr, by_visit):
    obs = tr.loc[tr.index.repeat(tr["n"]), ["L", "R", "B"]].reset_index(drop=True)
    sim = simulate(p, 400000, by_visit)
    scale = len(obs) / len(sim)
    k = range(len(CATS))
    print("\nObserved vs simulated record counts:")
    table = pd.DataFrame({"record": CATS})
    for col in ["L", "B"]:
        table[f"{col} obs"] = obs[col].value_counts().reindex(k, fill_value=0).to_numpy()
        table[f"{col} sim"] = (sim[col].value_counts().reindex(k, fill_value=0).to_numpy() * scale).round(0)
    print(table.to_string(index=False))
    print("\nRight-eye row % given left record, observed then simulated:")
    for name, d in [("observed", obs), ("simulated", sim)]:
        t = pd.crosstab(d["L"], d["R"]).reindex(index=k, columns=k, fill_value=0)
        t = (100 * t.div(t.sum(axis=1).replace(0, np.nan), axis=0)).round(0)
        t.index, t.columns = CATS, CATS
        print(f"\n{name}\n{t.to_string()}")


def typical_readings(p, n=1000000, seed=1):
    """Draws of unrounded typical readings, without the poor-vision component."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(p["mean_logmar"], p["sd_logmar"], n)
    if "w_typ2" in p:
        alt = rng.random(n) < p["w_typ2"]
        mu[alt] = rng.normal(p["mean_logmar"] + p["pop_shift"], p["s_typ2"], alt.sum())
    left = mu + rng.normal(0, p["sigma_eye"], n)
    right = mu + rng.normal(0, p["sigma_eye"], n)
    binoc = binocular_mean(mu, p) + rng.normal(0, p["sigma_bin"], n)
    if "lam_bin" in p:
        alt = rng.random(n) < p["lam_bin"]
        binoc[alt] = (binocular_mean(mu[alt], p) + p["bin_shift"]
                      + rng.normal(0, p.get("sigma_bin2", p["sigma_bin"]), alt.sum()))
    if "m_bin" in p:
        binoc = independent_binocular(p, rng, n)
    return {"monocular": left, "better-eye": np.minimum(left, right), "binocular": binoc}


def report(name, idata, tr, by_visit):
    names = [v for v in PARAMS + SPLITS if v in idata.posterior]   # h_bare is absent per line
    if "m_bin" in idata.posterior or EYES_ONLY:     # these two aren't used then
        names = [v for v in names if v not in ("sigma_bin", "delta_bin")]
    if "ds_eps" in idata.posterior:     # Ding & Sperling replaces the offset
        names = [v for v in names if v != "delta_bin"]
    summary = az.summary(idata, var_names=names, round_to=4)
    print(f"\n===== {name} =====")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    print("Per-chain mean_logmar:", idata.posterior["mean_logmar"].mean("draw").values.round(4))
    print("Per-chain sd_logmar:  ", idata.posterior["sd_logmar"].mean("draw").values.round(4))
    p = post_means(idata)
    print(f"Log-likelihood at posterior means: {loglik(p, tr, by_visit):.1f}")
    for name, y in typical_readings(p).items():
        print(f"Implied typical {name} reading: mean {y.mean():+.4f}, SD {y.std():.4f}, "
              f"share better than 20/15 {np.mean(y < L15):.3f}")
    sys.stdout.flush()
    return p


def main(tag):
    global EYES_ONLY
    EYES_ONLY = tag.startswith("eyes")
    idiff_max, by_visit, free_split = {
        "visit": (0.1, True, False), "reading": (0.1, False, False),
        "visit_idiff0.2": (0.2, True, False), "visit_split": (0.1, True, True),
        "reading_split": (0.1, False, True), "visit_split_idiff0.2": (0.2, True, True),
        "visit_lines": (0.1, True, "lines"), "visit_lines_bin": (0.1, True, "lines_bin"),
        "bin_ceiling": (0.1, True, "bin_ceiling"), "bin_20": (0.1, True, "bin_20"),
        "bin_slope": (0.1, True, "bin_slope"), "bin_style": (0.1, True, "bin_style"),
        "bin_mix": (0.1, True, "bin_mix"), "bin_mix_sd": (0.1, True, "bin_mix_sd"),
        "pop_mix": (0.1, True, "pop_mix"), "bin_indep": (0.1, True, "bin_indep"),
        "bin_indep_mix": (0.1, True, "bin_indep_mix"), "bin_indep_sep": (0.1, True, "bin_indep_sep"),
        "bin_indep_b1": (0.1, True, "bin_indep_b1"),
        "bin_indep_idiff0.2": (0.2, True, "bin_indep"),
        "bin_indep_sep_idiff0.2": (0.2, True, "bin_indep_sep"),
        "bin_indep_nofilter": (9.0, True, "bin_indep"),
        "bin_slope_idiff0.2": (0.2, True, "bin_slope"), "bin_slope_nofilter": (9.0, True, "bin_slope"),
        "eyes_idiff0.2": (0.2, True, "eyes"), "eyes_nofilter": (9.0, True, "eyes"),
        "eyes_mix_idiff0.2": (0.2, True, "eyes_mix"), "eyes_mix_nofilter": (9.0, True, "eyes_mix"),
        "eyes_wide_idiff0.2": (0.2, True, "eyes_wide"), "eyes_wide_nofilter": (9.0, True, "eyes_wide"),
        "ds_nofilter": (9.0, True, "ds")}[tag]
    tr = load_triples(idiff_max)
    print(f"{tr['n'].sum()} persons, {len(tr)} distinct record triples", flush=True)
    idata = fit(tr, by_visit, free_split)
    p = report(tag, idata, tr, by_visit)
    bare_meaning(p, tr, by_visit)
    check(p, tr, by_visit)
    idata.to_netcdf(HERE / f"person_records_{tag}.nc")


if __name__ == "__main__":
    # Run one variant per process, so the variants can run at the same time.
    main(sys.argv[1])
