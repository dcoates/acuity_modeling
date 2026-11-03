"""Does Ding & Sperling, passed through the recording rule, reproduce the binocular records?

Forward simulation, with no fitting to binocular records here:
  1. Draw unrounded left and right readings from the left/right reference fit (eyes_wide_nofilter).
  2. Combine them with Ding & Sperling Model 2.
  3. Add binocular reading noise.
  4. Record the result with the eyes' recording rule, in the same visit style as the eyes.
  5. Compare with observed binocular records, overall and jointly with the eye records.

The epsilon and gamma values come from fit_ding_sperling_typical.py, which used recorded values.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import fit_person_records as fpr

HERE = Path(__file__).parent
N = 2000000
SETTINGS = [("all ages 20-50", 7.03, 2.00), ("both eyes <= 0.3", 7.37, 3.92)]
NOISES = [0.0, 0.02, 0.046]
LINE_CAT = np.array([0, 3, 6, 8])


def record(y, line_style, p):
    cuts = np.array([p["b1"], p["b2"], p["b3"]])
    return np.where(line_style, LINE_CAT[np.searchsorted(cuts, y)],
                    np.searchsorted(fpr.edges(p, np.array)[3][:-1], y))


def ding_sperling(left, right, eps, gamma):
    k_l, k_r = 10.0 ** -left, 10.0 ** -right
    a_l, a_r = eps * k_l ** gamma, eps * k_r ** gamma
    return -np.log10(((1 + a_l) * k_l + (1 + a_r) * k_r) / (1 + a_l + a_r))


def simulate(p, eps, gamma, noise, rng):
    poor = rng.random(N) < p["w_poor"]
    mu = np.where(poor, rng.normal(p["mean_logmar"] + p["gap"], p["s_poor"], N),
                  rng.normal(p["mean_logmar"], p["sd_logmar"], N))
    left, right = mu + fpr.eye_noise(p, rng, N), mu + fpr.eye_noise(p, rng, N)
    binoc = ding_sperling(left, right, eps, gamma) + rng.normal(0, noise, N)
    line_style = rng.random(N) < p["p_line_style"]
    return pd.DataFrame({"L": record(left, line_style, p), "R": record(right, line_style, p),
                         "B": record(binoc, line_style, p), "y": binoc})


def main():
    p = fpr.post_means(az.from_netcdf(HERE / "person_records_eyes_wide_nofilter.nc"))
    tr = fpr.load_triples(9.0)
    obs = tr.loc[tr.index.repeat(tr["n"]), ["L", "R", "B"]].reset_index(drop=True)
    n = len(obs)
    k = range(len(fpr.CATS))
    table = pd.DataFrame({"observed": obs["B"].value_counts().reindex(k, fill_value=0).to_numpy()}, index=fpr.CATS)
    patterns = {
        "all three bare 20/15": lambda d: (d.L == 0) & (d.R == 0) & (d.B == 0),
        "all three bare 20/20": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 3),
        "eyes bare 20/20, B bare 20/15": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 0),
        "eyes bare 20/15, B bare 20/20": lambda d: (d.L == 0) & (d.R == 0) & (d.B == 3),
        "eyes bare 20/20, B 20/20 plus": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 2),
        "eyes bare 20/20, B 20/15 minus": lambda d: (d.L == 3) & (d.R == 3) & (d.B == 1),
    }
    joint = pd.DataFrame({"observed": {name: int(f(obs).sum()) for name, f in patterns.items()}})
    rng = np.random.default_rng(1)
    means = {}
    for label, eps, gamma in SETTINGS:
        for noise in NOISES:
            sim = simulate(p, eps, gamma, noise, rng)
            name = f"{label}, noise {noise}"
            table[name] = (sim["B"].value_counts().reindex(k, fill_value=0).to_numpy() * n / N).round(0).astype(int)
            joint[name] = [int(round(f(sim).mean() * n)) for f in patterns.values()]
            miss = np.abs(table[name] - table["observed"]).sum() / (2 * n)
            means[name] = (sim["y"].mean(), sim["y"].std(), miss)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("Binocular records\n", table.T.to_string())
    print("\nJoint with eye records\n", joint.T.to_string())
    print("\nImplied binocular reading before recording, and share of records misplaced")
    for name, (m, sd, miss) in means.items():
        print(f"  {name:36s} mean {m:+.3f}  SD {sd:.3f}  misplaced {miss:.3f}")


if __name__ == "__main__":
    main()
