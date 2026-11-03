# Binocular acuity modelling: handover notes

Status as of the last session. Read this before running anything. Code is in git; data and traces are not.

## Setup on a new machine

1. Python 3.10 (the project was run on 3.10.12).
2. `python3 -m venv ~/.venvs/binoc && ~/.venvs/binoc/bin/pip install -r requirements.txt`
3. Copy the input data files into this folder, through a secure channel (they contain MRNs):
   - `binoacuitydata.xlsx` (raw export, source of truth)
   - `nonstandard_stereo_sc.xlsx` (second VA block used for the sc rows)
4. Everything else is generated (see below).
5. Notebooks use the kernel "Python (binoc PyMC)", i.e. the venv above.

## Regenerate, in this order

```
~/.venvs/binoc/bin/python filter_valid_acuity.py          # -> binoacuitydata_valid.xlsx, nonint_age.xlsx, nonstandard_*.xlsx
~/.venvs/binoc/bin/python fit_ding_sperling_decimal_pymc.py   # -> ding_sperling_trace_decimal.nc (pooled model, summary notebook)
~/.venvs/binoc/bin/python refit_full.py                   # -> shared_eps, hier, learned_cutoff traces
~/.venvs/binoc/bin/python cv_cutoff_full.py               # -> ding_sperling_cutoff_full_cv.csv
~/.venvs/binoc/bin/python cv_sigma_interaction.py         # -> sigma_interaction_cv.csv, sigma_interaction_trace.nc
~/.venvs/binoc/bin/python sigma_delta_test.py             # -> sigma_delta_trace.nc (optional)
~/.venvs/binoc/bin/python fit_normal_mixture.py           # -> normal_mixture_trace_20_50_idiff0.1.nc (default settings)
~/.venvs/binoc/bin/python fit_healthy_truncated.py        # -> healthy_truncated_letter{0.01,0.095}.nc
```

Notes on running:
- The CV scripts are slow (many minutes each). The decimal fit takes a few minutes.
- `fit_joint_records.py` took about 26 minutes and failed (see below). Don't rerun it unless the model changes.
- `cv_power_mean.py` is degenerate. Don't rerun unless the parameter bounds change.
- Close any Jupyter kernel that has a trace file open before rerunning a fit. Open kernels lock the HDF5 files and make writes fail.
- The sensitivity runs for the mixture model (line window, letter window, idiff cutoffs) were done by importing `fit_normal_mixture` and setting its module constants; the script itself runs only the default settings.

## Data rules (current)

- Key: column A `MRN` in all three source files. Kept as a row key.
- Raw acuity: `raw_left`, `raw_right`, `raw_both` = the three VA columns as written. Main file: `Last_VA_CC_-_*`; sc file: the `.1` block (`Last VA CC - *.1`); dash/dot-age rows: `Last VA CC - *`.
- Ages: integers, plus single-digit ages written as `8-` or `6.` (= 8 or 6, confirmed by the user). Stored in `age`.
- Dropped: "Dc" rows, rows with eye logMAR ≥ 1.1 (decimal k ≤ 0), rows where raw text is neither `20/N` nor `20/N±k` (e.g. trailing dash).
- `logmar_better` = lower of left/right; `idiff` = log10(k_better/k_worse) = logmar_worse − logmar_better; `advantage` = logmar_better − logmar_binocular.
- Use `load_all_df()` in `fit_ding_sperling_decimal_pymc.py` for the combined table. `load_all_rows()` returns the three arrays used by the fits.
- Data quality: five rows with large negative `advantage` were flagged as possible entry errors (find them with `advantage` in `load_all_df()`). The user was checking the raw file.

## Model results so far

**Ding & Sperling Model 2 (decimal k, pooled):** fits well (RMSE 0.0585, bias +0.002, 7,085 rows). Current reference for the binocular prediction.

**Learned cutoff:** cutoff ≈ 0.23 logMAR (only on high-vision rows). Beats fixed cutoffs for poor vision, but not the best fixed cutoff over all rows (0.15). The 0.4 cutoff in older scripts was arbitrary; don't use it.

**Interaction model (learned cutoff + Δ-dependent spread):** best Ding-family model by CV (+0.021 overall, +0.077 poor vision, +0.043 for >0.25). κ ≈ 1.26. The mean prediction is unchanged; the Δ effect is only on spread in good/middle vision.

**Power-mean (Legge-like):** degenerate (cutoff at the prior edge, p ≈ 15 and 1.8). Not reliable.

**Shared-ε and hierarchical (refit_full.py):** still use the arbitrary 0.4 group indicator. Treat as superseded.

**Normal recording mixture (fit_normal_mixture.py):** latent better-eye logMAR ~ Normal(m, s); records are line-only or letter form, with a recording rule. Results (ages 20–50, idiff ≤ 0.1):
- Mean about −0.037 (roughly 20/18), SD about 0.064, line-only share about 0.66.
- Your target (mean −0.086, SD 0.095) is not supported.
- Fitted SD depends on the line window (0.067 at ±0.025, 0.059 at ±0.095).

**Healthy-only truncated fit (fit_healthy_truncated.py, records ≤ 0.10):** mean −0.040, SD 0.055 (letter window ±0.01). The ±0.095 letter window collapses the SD to about 0.02, so it is not usable.

**Within-row sign agreement (letter modifiers):** signs agree more often than chance (better vs worse 0.81 against 0.68 by chance; better vs binocular 0.88 against 0.81). A shared recording rule looks plausible.

**Interval-censored Normal (fit_interval_censored.py, better eye only):** bare records as between-neighbouring-lines gives mean −0.044, SD 0.051, but the record sets overlap and the mean moves from −0.027 to −0.079 with the record assumption. Better-eye data alone can't settle it.

**Per-person record model (fit_person_records.py, run one variant per process: `python fit_person_records.py visit_split`):** one latent acuity per person, all three readings, recording style per visit. Reference variant is `visit_split` (R-hat ≤ 1.003, 0 divergences):
- Typical person mean −0.039, SD 0.074; implied better-eye reading mean −0.052, SD 0.076. Target (−0.086, 0.095) still not reached. Means across converged variants: −0.036 to −0.048.
- About 32% of visits are line-only style, where bare 20/20 spans −0.15 to +0.09 (roughly the ±1 line idea). The rest are letter style, where bare means ±0.021 (about one letter).
- Binocular offset −0.039 (about 2 letters), so the binocular record can't be a plain third reading.
- Letter style favours "−" records: split points are −0.041 and +0.071, not the line midpoints.
- Style per visit fits far better than per reading. `reading_split` did not converge (one chain in a different mode); don't use it.
- Remaining misfit: 20/20− (467 observed, 310 simulated), bare 20/25 (65 vs 135).

**Per-line windows (`visit_lines`, now the best fit; R-hat ≤ 1.013, 0 divergences, log-likelihood +160 for 3 extra parameters):** the bare rule is not the same at each line. This variant caps the poor-vision share at 5%, because otherwise some chains let that component swallow the typical population.
- Letter visits, bare window: 20/15 up to +0.01 worse (at the prior floor), 20/20 from −0.041 to +0.013, 20/25 ±0.005 (few records).
- Line-only visits: bare 20/20 runs from −0.174 to +0.069, bare 20/25 from +0.069 to +0.119.
- Shared across lines: "−" is preferred over "+" (splits at −0.058 and +0.081).
- Typical person mean −0.051, SD 0.076; implied better-eye reading mean −0.065, SD 0.078; implied binocular mean −0.092, SD 0.088.

Left and right eyes share a rule: their record table is symmetric (Bowker p = 0.18 at idiff ≤ 0.1; p = 0.016 with no idiff filter, leaning slightly to the right eye around 20/15–20/20).

**Reference fit is now `bin_indep`** (see "Why does binocular differ?" below): one shared rule, binocular independent. person_model_distributions.ipynb and the record_types_monocular/binocular figures use it. `record_types_by_reading` is out of date.

**Binocular's own rule (`visit_lines_bin`, set `BINOC_CHAINS=20` for one run alone; R-hat ≤ 1.010, 0 divergences, log-likelihood +109 for 9 extra parameters):** binocular records do not follow the eye rule around 20/15–20/20; near 20/25 the two rules agree within uncertainty.
- Line-only visits: bare 20/20 starts at −0.146 for binocular, but −0.238 for eyes, near the prior limit of −0.25, so line-only eyes are almost never bare 20/15.
- Letter visits: bare 20/20 better side is 0.046 for binocular (near the 0.05 limit), 0.015 for eyes.
- Estimates move with this choice: typical person mean −0.031, SD 0.066; better eye −0.043, SD 0.068; binocular −0.080, SD 0.077. Offset −0.049.

**Why does binocular differ? (log-likelihood gain over shared-rule `visit_lines`, −11345.1):** the user's view is that the quantizing rule should be the same, because it is a function of the measurement.
- `bin_slope` (binocular scales with acuity, slope 0.96): +2.2 for 1 parameter. Not worth it.
- `bin_style` (fewer binocular letter records, xi 0.03): +6.7 for 1 parameter. Small.
- `bin_ceiling` (only boundaries around 20/15 differ): +41 for 3.
- `bin_20` (only 20/20 letter windows differ): +53 for 2.
- `visit_lines_bin` (all differ): +109 for 9.
- **Independent binocular (user's preferred framing, no gain or summation assumed):** `bin_indep` gives binocular its own distribution and poor-vision component; only the visit's recording style links it to the eyes. Shared rule −12404.5; separate rule `bin_indep_sep` −12349.4, +55 for 9 parameters, far less than the +109 in the linked fit. Under the shared rule the binocular poor-vision component goes to its limits (share 0.046, located at about +0.01), so it is being used as extra mass near 20/20. A rule difference and a non-Normal binocular shape are not separated. `bin_indep_b1` (only `b1` differs, run before binocular had its own poor-vision component) added nothing. `bin_indep_mix` did not converge.
- Independent estimates, shared rule: monocular −0.048 / 0.075, better eye −0.060 / 0.074, binocular −0.098 / 0.088. Separate rule: −0.034 / 0.069, −0.045 / 0.068, −0.071 / 0.068.
- So the ceiling alone doesn't explain it. The main unexplained shared-rule misfit is too many binocular 20/20+ (147–156 vs 68) and too few bare 20/20 (about 1,170 vs 1,285). A non-Normal binocular distribution is the untested option that would keep one rule.

## What doesn't work

- **Letter-sign pattern:** no single recording rule reproduces the observed share of "worse" (−) letter records on each line. Observed: 20/15 0.99, 20/20 0.75, 20/25 0.50 (20/25 has only 100 records in the subset). The models predict 0.88, 0.28 and 0.04 for the healthy-only fit.
- **Wide letter window (±0.095):** absorbs all records. Line-only share goes to zero.
- **20/15 ceiling (cap at −0.125):** capped fits did not converge (R-hat up to 1.7).
- **Joint better-eye + binocular model (fit_joint_records.py):** letter window went to ±0.30, SD 0.15, 8 divergences. Not usable.

## Open questions (need the user / clinic)

1. What does a bare `20/20` (no letters) mean? "At least 20/20", a default, or a real reading? It was many different clinicians and students, so it may not be consistent.
2. Is 20/15 the top line on the charts (a ceiling)? The data show almost no "+" letters at 20/15.
3. Should the poor-vision eyes be dropped (excluded) or modelled as a separate component? The user leaned toward excluding them for now.
4. Should the binocular record be a third reading of the same person's acuity (no summation offset)? That was the chosen simpler option; summation (Ding–Sperling) is a separate question and isn't being tested here.

## Suggested next steps

- Per-line empirical model: estimate the letter distribution separately for each line from the data, with no window assumption. Test whether line-only records alone look Normal.
- Grouping by clinician or site isn't possible: the data are from one site and clinicians aren't known. Recording style per visit stands in for this in the per-person model.
- On Windows the venv's Python is `~/.venvs/binoc/Scripts/python.exe`. g++ is absent there, so use `nuts_sampler="nutpie"`.
