"""Grid search for the split cutoff, by held-out fit.

For each candidate cutoff c, fit the shared-eps model (one eps; separate gamma and sigma for
x <= c and x > c, with x = better-eye logMAR) and score it by 5-fold CV.

Evaluation rows are fixed across candidates: all rows with x > 0.4 (the high-acuity range),
so every cutoff is scored on the same held-out data. Training uses all other rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows
from cv_shared_eps import fit_shared_eps, predict_shared_eps

HERE = Path(__file__).parent
CANDIDATES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
EVAL_CUT = 0.4
N_FOLDS = 5
SEED = 0


def main():
    k_better, k_worse, y = load_all_rows()
    lo_all, up_all = bin_edges(y)
    x = -np.log10(k_better)
    eval_idx = np.where(x > EVAL_CUT)[0]
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(eval_idx), N_FOLDS)

    rows = []
    for c in CANDIDATES:
        lpd_all = np.zeros(len(y)); pred_all = np.zeros(len(y))
        group = (x > c).astype(int)
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(len(y)), test)
            idm = fit_shared_eps(k_better[train], k_worse[train], lo_all[train], up_all[train],
                                 group[train], seed=100 + f)
            lpd, pr = predict_shared_eps(idm, k_better[test], k_worse[test], lo_all[test],
                                         up_all[test], group[test])
            lpd_all[test] = lpd; pred_all[test] = pr
        r = y[eval_idx] - pred_all[eval_idx]
        row = {"cutoff": c,
               "mean_log_pred_density": lpd_all[eval_idx].mean(),
               "RMSE": np.sqrt(np.mean(r**2)),
               "bias": r.mean(),
               "n_high_regime_train": int((group == 1).sum())}
        rows.append(row)
        print(f"cutoff {c:.2f}: mean log pred. density = {row['mean_log_pred_density']:.3f}, "
              f"RMSE = {row['RMSE']:.4f}, bias = {row['bias']:+.4f}", flush=True)

    out = pd.DataFrame(rows)
    out_path = HERE / "ding_sperling_cutoff_grid.csv"
    out.to_csv(out_path, index=False)
    best = out.loc[out.mean_log_pred_density.idxmax()]
    print(f"\nBest cutoff by held-out fit: {best.cutoff:.2f} "
          f"(mean log pred. density {best.mean_log_pred_density:.3f})")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
