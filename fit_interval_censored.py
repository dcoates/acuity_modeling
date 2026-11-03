"""Normal population of typical eyes, with bare line records treated as interval-censored.

Latent: true better-eye logMAR, mu ~ Normal(m, s), truncated above at T (poor eyes excluded).
A bare record such as "20/20" says only that mu lies between the neighbouring lines.

Three record models, all on ages 20-50 with idiff <= 0.1:
  censored:  bare = between the neighbouring lines; letter = exact, +/-LETTER_HW.
             Likelihood is the product of P(mu in record's set); the sets overlap.
  ordinal:   as censored, but a letter record says only which side of its line mu is on.
  cells:     generative. Lines cut mu into non-overlapping cells. With probability pi the
             person gets a bare record, at the better end of the cell with probability q,
             else the worse end. With probability 1 - pi the record is a letter value in
             that cell. Everyone better than 20/15 gets a bare 20/15 (chart ceiling).

Records are aggregated to counts, so sampling is fast. Prints per-record observed and
predicted counts for the cells model.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from fit_ding_sperling_decimal_pymc import load_all_df

HERE = Path(__file__).parent
AGE_LO, AGE_HI = 20, 50
IDIFF_MAX = 0.1
LINES = np.array([10, 15, 20, 25, 30, 40, 50])
ELL = np.log10(LINES / 20)
TOP = 1               # index in LINES of 20/15, the chart ceiling in the cells model
FLOOR = -1.0          # finite lower bound for "better than the top line"
LETTER_HW = 0.01
CORES = 8             # chains run in parallel, one core each


def load_records():
    df = load_all_df()
    df = df[df["age"].between(AGE_LO, AGE_HI) & (df["idiff"] <= IDIFF_MAX)]
    raw = pd.Series(np.where(df["left"] <= df["right"], df["raw_left"], df["raw_right"]),
                    index=df.index).astype(str).str.strip()
    ex = raw.str.extract(r"^20/(\d+)\s*([+-]?)\s*(\d*)$")
    out = pd.DataFrame({"line": ex[0].astype(int), "sign": ex[1], "value": df["logmar_better"].round(3)})
    return out[out["line"].isin(LINES)].groupby(["line", "sign", "value"]).size().rename("n").reset_index()


def line_index(line):
    return int(np.flatnonzero(LINES == line)[0])


def select(rec, t_line):
    """Keep records whose information lies above the top line and below the truncation."""
    t = np.log10(t_line / 20)
    bare = (rec["sign"] == "") & (rec["line"] >= 15) & (rec["line"] < t_line)
    letter = (rec["sign"] != "") & (rec["value"] < t) & (rec["value"] > ELL[TOP])
    return rec[bare | letter].reset_index(drop=True), t


def censored_sets(rec, ordinal):
    lo, hi = [], []
    for row in rec.itertuples():
        i = line_index(row.line)
        if row.sign == "":
            lo.append(ELL[i - 1])
            hi.append(ELL[i + 1])
        elif not ordinal:
            lo.append(row.value - LETTER_HW)
            hi.append(row.value + LETTER_HW)
        elif row.sign == "+":
            lo.append(ELL[i - 1])
            hi.append(ELL[i])
        else:
            lo.append(ELL[i])
            hi.append(ELL[i + 1])
    return np.array(lo), np.array(hi)


def letter_bins(rec):
    """Partition each cell between lines into bins, one per distinct letter value."""
    values = rec.loc[rec["sign"] != "", "value"].to_numpy()
    bins = {}
    for a, b in zip(ELL[TOP:-1], ELL[TOP + 1:]):
        v = np.unique(values[(values > a) & (values < b)])
        edges = np.r_[a, (v[1:] + v[:-1]) / 2, b]
        bins.update({x: (edges[j], edges[j + 1]) for j, x in enumerate(v)})
    return bins


def cdf(x, m, s):
    return 0.5 * (1.0 + pt.erf((x - m) / (s * np.sqrt(2.0))))


def sample(model):
    with model:
        return pm.sample(draws=2000, tune=1500, chains=CORES, cores=CORES, target_accept=0.95,
                         nuts_sampler="nutpie", random_seed=1, progressbar=False)


def fit_censored(rec, t, ordinal):
    lo, hi = censored_sets(rec, ordinal)
    n = rec["n"].to_numpy()
    with pm.Model() as model:
        m = pm.Normal("mean_logmar", mu=-0.086, sigma=0.1)
        s = pm.HalfNormal("sd_logmar", sigma=0.2)
        z = cdf(t, m, s)
        p = pt.clip((cdf(hi, m, s) - cdf(lo, m, s)) / z, 1e-12, 1.0)
        pm.Potential("record_loglik", pt.sum(n * pt.log(p)))
    return sample(model)


def cells_probs(rec, t, m, s, pi, q, phi):
    """Record probabilities in the cells model; phi is the Normal CDF for the backend in use."""
    bins = letter_bins(rec)
    z = phi(t, m, s)

    def mass(a, b):
        return (phi(min(b, t), m, s) - phi(a, m, s)) / z

    out = []
    for row in rec.itertuples():
        if row.sign != "":
            a, b = bins[row.value]
            out.append((1 - pi) * mass(a, b))
            continue
        i = line_index(row.line)
        below = mass(ELL[i], ELL[i + 1])
        above = mass(ELL[i - 1], ELL[i])
        if i == TOP:
            out.append(pi * q * below + mass(FLOOR, ELL[TOP]))
        else:
            out.append(pi * (q * below + (1 - q) * above))
    return out


def fit_cells(rec, t):
    n = rec["n"].to_numpy()
    with pm.Model() as model:
        m = pm.Normal("mean_logmar", mu=-0.086, sigma=0.1)
        s = pm.HalfNormal("sd_logmar", sigma=0.2)
        pi = pm.Beta("p_bare", alpha=1.0, beta=1.0)
        q = pm.Beta("q_better_end", alpha=1.0, beta=1.0)
        p = pt.stack(cells_probs(rec, t, m, s, pi, q, cdf))
        pm.Potential("record_loglik", pt.sum(n * pt.log(pt.clip(p, 1e-12, 1.0))))
    return sample(model)


def report(name, idata, var_names):
    summary = az.summary(idata, var_names=var_names, round_to=4)
    print(f"\n===== {name} =====")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}", flush=True)


def check_cells(rec, t, idata):
    post = idata.posterior
    args = [float(post[v].mean()) for v in ["mean_logmar", "sd_logmar", "p_bare", "q_better_end"]]
    phi = lambda x, m, s: norm.cdf(x, m, s)
    p = np.array(cells_probs(rec, t, *args, phi), dtype=float)
    table = rec.assign(predicted=(p * rec["n"].sum() / p.sum()).round(1))
    print("\nObserved n vs predicted, cells model:")
    print(table.to_string(index=False))
    print(f"Probabilities sum to {p.sum():.3f} (below 1 where dropped records take mass)")


def main():
    rec = load_records()
    print(f"{rec['n'].sum()} records on standard lines, {len(rec)} distinct")
    for t_line in [30, 40]:
        sub, t = select(rec, t_line)
        print(f"\n########## truncation at 20/{t_line} ({t:.3f}), {sub['n'].sum()} records ##########")
        names = ["mean_logmar", "sd_logmar"]
        for name, ordinal in [("censored", False), ("ordinal", True)]:
            idata = fit_censored(sub, t, ordinal)
            report(name, idata, names)
            idata.to_netcdf(HERE / f"interval_{name}_t{t_line}.nc")
        idata = fit_cells(sub, t)
        report("cells", idata, names + ["p_bare", "q_better_end"])
        idata.to_netcdf(HERE / f"interval_cells_t{t_line}.nc")
        check_cells(sub, t, idata)


if __name__ == "__main__":
    main()
