"""How binocular compares with the better eye when the eyes differ, by better-eye acuity.

Ages 20-50, all valid rows. Recorded logMAR values (letters included). Prints, for each band of
interocular difference and better-eye acuity, how often binocular is worse than, equal to, or
better than the better eye, and the mean binocular minus better-eye value.
"""

import numpy as np
import pandas as pd

from fit_ding_sperling_decimal_pymc import load_all_df


def main():
    df = load_all_df()
    df = df[df["age"].between(20, 50)].copy()
    df["gap"] = df["logmar_binocular"] - df["logmar_better"]
    df["diff_band"] = pd.cut(df["idiff"], [-0.01, 0.0001, 0.05, 0.1, 0.2, 0.3, 0.5, 3],
                             labels=["0", "0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", ">0.5"])
    df["better_band"] = pd.cut(df["logmar_better"], [-1, 0.0001, 0.1, 0.3, 3],
                               labels=["20/20 or better", "to 20/25", "to 20/40", "worse than 20/40"])
    tol = 1e-6
    summary = df.groupby(["better_band", "diff_band"], observed=True).agg(
        n=("gap", "size"),
        mean_binoc_minus_better=("gap", "mean"),
        share_worse=("gap", lambda g: (g > tol).mean()),
        share_equal=("gap", lambda g: (g.abs() <= tol).mean()),
        share_better=("gap", lambda g: (g < -tol).mean()),
    )
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    good = df[(df["logmar_better"] <= 0.1) & (df["idiff"] > 0.1)]
    print(f"\nGood better eye (<= 20/25), eyes differ by > 0.1: n={len(good)}, "
          f"binocular worse than better eye in {np.mean(good['gap'] > tol):.2f}, "
          f"mean binocular minus better {good['gap'].mean():+.3f}")
    close = df[(df["logmar_better"] <= 0.1) & (df["idiff"] <= 0.05)]
    print(f"Good better eye, eyes within 0.05: n={len(close)}, binocular worse in {np.mean(close['gap'] > tol):.2f}, "
          f"mean {close['gap'].mean():+.3f}")


if __name__ == "__main__":
    main()
