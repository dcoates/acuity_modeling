"""Full-data refits (no CV) for the models used in the summary notebook.

Writes, on all current rows (main valid file + sc file + dash/dot-age rows):
  ding_sperling_trace_shared_eps.nc
  ding_sperling_trace_hier.nc
  ding_sperling_trace_learned_cutoff.nc

The pooled decimal, split, and sigma-delta fits have their own scripts and are run separately.
"""

import numpy as np
import arviz as az

from fit_ding_sperling_discrete_pymc import bin_edges
from fit_ding_sperling_decimal_pymc import load_all_rows
from cv_shared_eps import fit_shared_eps
from cv_split_hierarchical import fit as fit_hier
from cv_learned_cutoff import fit_learned
from pathlib import Path

HERE = Path(__file__).parent
CUTOFF = 0.4


def main():
    k_better, k_worse, y = load_all_rows()
    lo, up = bin_edges(y)
    x = -np.log10(k_better)
    high = (x > CUTOFF).astype(int)
    print(f"Rows: {len(y)}  (high-group rows, for the group-indicator models: {high.sum()})", flush=True)

    print("\n=== shared eps, separate gamma/sigma ===", flush=True)
    idm = fit_shared_eps(k_better, k_worse, lo, up, high, seed=1)
    s = az.summary(idm, var_names=["eps", "gamma_low", "gamma_high", "sigma_low", "sigma_high"], round_to=4)
    print(s.to_string(), flush=True)
    idm.to_netcdf(HERE / "ding_sperling_trace_shared_eps.nc")

    print("\n=== hierarchical, shared eps and gamma ===", flush=True)
    idh = fit_hier(k_better, k_worse, lo, up, high, hier=True, seed=1)
    s = az.summary(idh, var_names=["eps", "gamma", "sigma_low", "sigma_high"], round_to=4)
    print(s.to_string(), flush=True)
    idh.to_netcdf(HERE / "ding_sperling_trace_hier.nc")

    print("\n=== learned cutoff ===", flush=True)
    idl = fit_learned(k_better, k_worse, lo, up, x, seed=1)
    s = az.summary(idl, var_names=["eps", "gamma_low", "gamma_high", "sigma_low", "sigma_high", "cutoff", "tau"], round_to=4)
    print(s.to_string(), flush=True)
    idl.to_netcdf(HERE / "ding_sperling_trace_learned_cutoff.nc")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
