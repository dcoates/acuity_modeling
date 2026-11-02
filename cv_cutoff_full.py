"""Cutoff comparison with evaluation rows that do not depend on any cutoff.

5-fold CV over ALL rows (the same folds for every candidate). Each candidate predicts the
held-out rows; the score is the mean log predictive density over all held-out rows, and also
reported on the subset with better-eye logMAR > 0.4 and > 0.25 (for reference only).

Candidates
  fixed c in CANDIDATES: shared-eps model, separate gamma and sigma for x <= c and x > c
  learned:               soft learned-cutoff model (cutoff and width learned in each fold)

x = better-eye logMAR. Nothing in the evaluation rows depends on the cutoff.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows
from cv_shared_eps import fit_shared_eps, predict_shared_eps
from cv_learned_cutoff import fit_learned, predict_learned

HERE = Path(__file__).parent
CANDIDATES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_FOLDS = 5
SEED = 0


def cv_scores(name, fit_predict):
    """fit_predict(train_idx, test_idx) -> (lpd, pred) for the test rows."""
    lpd_all = np.zeros(len(Y)); pred_all = np.zeros(len(Y))
    for f, test in enumerate(FOLDS):
        train = np.setdiff1d(np.arange(len(Y)), test)
        lpd, pr = fit_predict(train, test, f)
        lpd_all[test] = lpd; pred_all[test] = pr
    row = {"model": name}
    for label, mask in [("all", np.ones(len(Y), bool)), ("x>0.4", X > 0.4), ("x>0.25", X > 0.25)]:
        r = Y[mask] - pred_all[mask]
        row[f"lpd_{label}"] = lpd_all[mask].mean()
        row[f"rmse_{label}"] = np.sqrt(np.mean(r**2))
        row[f"bias_{label}"] = r.mean()
    print(f"{name:<14} all: lpd {row['lpd_all']:.3f} | x>0.4: lpd {row['lpd_x>0.4']:.3f} "
          f"bias {row['bias_x>0.4']:+.4f} | x>0.25: lpd {row['lpd_x>0.25']:.3f}", flush=True)
    return row


def main():
    global Y, X, FOLDS
    k_better, k_worse, Y = load_all_rows()
    lo_all, up_all = bin_edges(Y)
    X = -np.log10(k_better)
    rng = np.random.default_rng(SEED)
    FOLDS = np.array_split(rng.permutation(len(Y)), N_FOLDS)

    rows = []
    for c in CANDIDATES:
        group = (X > c).astype(int)

        def fp(train, test, f, c=c, group=group):
            idm = fit_shared_eps(k_better[train], k_worse[train], lo_all[train], up_all[train],
                                 group[train], seed=100 + f)
            return predict_shared_eps(idm, k_better[test], k_worse[test], lo_all[test],
                                      up_all[test], group[test])
        rows.append(cv_scores(f"fixed {c:.2f}", fp))

    def fp_learned(train, test, f):
        idm = fit_learned(k_better[train], k_worse[train], lo_all[train], up_all[train],
                          X[train], seed=200 + f)
        return predict_learned(idm, k_better[test], k_worse[test], lo_all[test],
                               up_all[test], X[test])
    rows.append(cv_scores("learned", fp_learned))

    out = pd.DataFrame(rows)
    out_path = HERE / "ding_sperling_cutoff_full_cv.csv"
    out.to_csv(out_path, index=False)
    fixed = out[out.model.str.startswith("fixed")]
    best = fixed.loc[fixed["lpd_all"].idxmax()]
    print(f"\nBest fixed cutoff over all rows: {best.model} (lpd {best.lpd_all:.3f})")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
