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
- The user's target (mean −0.086, SD 0.095) is for **binocular** acuity, not the better eye, so this better-eye comparison doesn't test it. The 2D Ding & Sperling fit (`fit_person_eyes2d.py`) gives typical binocular −0.085, SD 0.076.
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

**`bin_indep` is the fit used by person_model_distributions.ipynb and the record_types_monocular/binocular figures, but it has a known flaw:** independence can't reproduce how closely binocular records follow the eyes. All three bare 20/15: 381 observed, 183 modelled. The line-only boundary `b1` goes to its −0.25 limit to compensate, and stays there at idiff ≤ 0.2 and with no filter, so the idiff filter is not the main cause.

`bin_slope` is the same shared rule with a free binocular mean, SD and correlation (0.85), so it assumes no particular gain. It reproduces those visits (382 vs 381; all bare 20/20 788 vs 781), with log-likelihood −11342.9 against −12404.5 for `bin_indep` on the same data. Monocular −0.050 / 0.079, binocular −0.090 / 0.085, `b1` −0.169. Its remaining misfits are one eye bare 20/15 with the other bare 20/20 (45 vs 0, removed by the idiff filter) and binocular 20/20+ (156 vs 68). Compared with `bin_slope`, a separate binocular rule (`visit_lines_bin`) gains about +107 for 8 extra parameters, so the +55 evidence below understates the rule question.

Correlated fit by filter (idiff ≤ 0.1 / ≤ 0.2 / none): `b1` −0.169 / −0.152 / −0.137, so the filter does affect it; monocular mean −0.050 / −0.049 / −0.047, SD 0.079 / 0.080 / 0.081; binocular mean −0.090 / −0.090 / −0.088, SD 0.085 / 0.084 / 0.083; correlation 0.85 / 0.87 / 0.88; `sigma_eye` 0.023 / 0.033 / 0.039. One eye bare 20/15 with the other bare 20/20 is over-predicted once visible (113 vs 69 at ≤ 0.2, 183 vs 69 with no filter), so a single Normal for eye differences is too simple: `sigma_eye` grows to absorb a few large differences.

**Left and right only (`eyes_*`, binocular left out of the likelihood):**
- One Normal for eye differences: idiff ≤ 0.2 gives mean −0.045, reading SD 0.076, `b1` −0.170; no filter gives −0.054, 0.090, `b1` −0.230. Bare + bare patterns fit well (one eye bare 20/15 and the other bare 20/20: 69 vs 69), so the over-prediction above came from including binocular. `b1` stays past 20/15 without binocular, so binocular independence wasn't its only cause.
- Two-part eye differences (`eyes_mix_*`): log-likelihood +117 at idiff ≤ 0.2 and +272 with no filter for 2 parameters, and it makes the filter matter far less (mean −0.038 / −0.042, person SD 0.055 / 0.058, reading SD about 0.063 / 0.070). But its limits are hit: `sigma_eye` at 0.005 (floor), `u_eye` 0.28 near its 0.3 cap at idiff ≤ 0.2. Most visits have nearly identical eye readings (L-R correlation 0.99 among them), which may be real or clinician anchoring. Not separated. Still misses bare 20/20 + 20/20 minus (about 180 vs 319).

- With wide limits and a 0.002 grid (`eyes_wide_*`, set `BINOC_GRID=0.002`), no limits are hit. No filter: mean −0.041, person SD 0.058, reading SD 0.070, small-part SD 0.007, 76% of eyes small, large-part SD 0.080. Nearly unchanged from `eyes_mix_nofilter`, so that fit was sound. idiff ≤ 0.2: mean −0.030, person SD 0.051, reading SD 0.058, small-part SD 0.003, 68% small. The small-part SD is well under one letter (0.02), so it is too small to be test–retest noise. Clinicians recording the second eye like the first is the likely reading, but this is untested. `eyes_wide_nofilter` is the suggested left/right reference.

person_model_distributions.ipynb and `record_types_monocular` now use `eyes_wide_nofilter` (left and right only, no filter). `record_types_binocular` and `record_types_by_reading` are out of date.

**Ding & Sperling Model 2 on ages 20–50 (`fit_ding_sperling_typical.py`, recorded values, no recording model):**
- All 3,605 rows: ε 7.03, γ 2.00, σ 0.046, RMSE 0.046, bias −0.002.
- idiff ≤ 0.2 (3,509): ε 7.07, γ 2.16, RMSE 0.046.
- Both eyes ≤ 0.3 (3,483): ε 7.37, γ 3.92, σ 0.042, RMSE 0.043.
- ε is stable; γ is not, because it is set mostly by the few poorer eyes. The mean binocular value is predicted well in all three (bias under 0.002).
- Through the recording rule (`ding_sperling_through_rule.py`): simulated left/right readings from `eyes_wide_nofilter`, combined by Ding & Sperling and recorded with the eyes' rule and visit style, do not match binocular records. There are too few bare 20/15 (905–1,022 vs 1,227) and two to three times too many 20/20+ (176–246 vs 73). Eyes bare 20/20 with binocular bare 20/15 is 0–73 vs 202, because `b1` at −0.227 leaves almost no binocular readings below it on line-only visits. Implied binocular reading mean −0.061 to −0.063. Best case (noise 0.046) misplaces 8.6% of records. ε and γ were fitted to recorded values, so the 20/15 ceiling may make them understate summation.

**Ding & Sperling inside the record model (`ds_nofilter`, `BINOC_CHAINS=20 BINOC_GRID=0.002`, 34 min):** left and right with two-part differences, binocular = Ding & Sperling of the person value (both eyes taken as equal, so binocular noise absorbs unequal eyes), one shared rule, no filter. R-hat ≤ 1.009, no divergences, log-likelihood −13512.0 (−13846.4 for `bin_slope_nofilter` on the same data).
- `b1` −0.141 (−0.158 to −0.120), close to 20/15 for the first time. The eye small-part SD is 0.013 (82% of eyes), no longer implausibly small.
- ε 5.3, γ 0.28 (0.02 to 0.70): binocular gain is nearly constant, about 0.035 logMAR at the mean. γ is far lower than the 2.0 fitted to recorded values.
- Typical monocular −0.049 / 0.069, better eye −0.056 / 0.069, binocular −0.084 / 0.076.
- Binocular records misplaced 5.3% (8.6% at best for the earlier forward simulation). Remaining misfits: binocular 20/20+ 161 vs 73; one eye bare 20/15 and the other bare 20/20 149 vs 69; eyes bare 20/20 + 20/20 minus 175 vs 319.

**Each eye's own acuity, with Ding & Sperling (`fit_person_eyes2d.py`, 2D grid of left/right true acuity at 0.01 steps, 20 chains, 74 min):** true acuity = person value + eye difference (two parts); readings add test noise; binocular = Ding & Sperling of the two eyes' true acuities, plus noise. One shared rule, style per visit, no filter. R-hat ≤ 1.006, no divergences, log-likelihood −13504.8 (−13512.0 for the shortcut `ds_nofilter`).
- ε 4.77, γ 1.87 (1.46 to 2.25), close to the γ of 2.0 fitted to recorded values. The shortcut's γ of 0.28 came from feeding Ding & Sperling the same value for both eyes.
- `b1` −0.140 (−0.156 to −0.123), consistent with the 20/15 line.
- Typical monocular −0.048 / 0.078, better eye −0.065 / 0.076, binocular −0.085 / 0.076. Binocular gain over the better eye 0.020.
- Test noise 0.011 is at its 0.010 floor (set by the grid step), and the small eye difference is 0.004 (80% of eyes), so the eyes are still recorded very alike. `h20_better` 0.046 is near its 0.05 limit.
- Binocular records misplaced 4.9%. Same remaining misfits as before: binocular 20/20+ 163 vs 73; one eye bare 20/15 and the other bare 20/20 150 vs 69; eyes bare 20/20 + 20/20 minus 176 vs 319.

**Poor vision and large interocular differences included (`fit_person_eyes2d_full.py`, 20 chains, 95 min):** grid to +1.4 (0.01 steps to 0.4, then 0.03), each line 20/30 to 20/400 its own category (21 categories), no filter, poor-vision component. Did **not converge**: R-hat up to 1.24 (`b2`, `b3`), 1.16 (`u_eye`), ESS as low as 59; 0 divergences. Treat as a rough reference. Log-likelihood −14242.7. The wide eye part (σ_d2 0.31, 6% of eyes) absorbs the poor eyes, so "typical" monocular SD is 0.11. γ 1.01. Eyes differing by > 0.2: binocular minus better-eye record obs −0.007, model +0.028 (too much binocular loss). `check_unequal_eyes.py` gives the observed patterns: good better eye (≤ 20/25) with idiff > 0.1, binocular worse than the better eye in 25%, mean +0.009; within 0.05, worse in 10%, mean −0.017; poorer better eyes gain about 0.08–0.09 when the eyes are equal.

**One-eye-worse component + sigmoid γ (`fit_person_eyes2d_ext.py asym+sig`, 20 chains, 154 min):** chains split into three modes. Per-chain log density: mode A (15 chains) −14230, a small poor-eye tail (v 0.012, shift 0.5); mode B (chains 1 and 17) **−14195**; mode C (3 chains) −14239, `b2` at its 0.15 limit. Pooled R-hat up to 1.43, so the pooled summary is meaningless. Mode B alone (2 chains, R-hat ≤ 1.006, 0 divergences), log-likelihood at posterior means −14116.7 (+126 over the baseline):
- Eye differences come in three kinds: 69% near identical (σ 0.008), **25% one eye worse by 0.060 ± 0.034** (about three letters, eye at random), 5.6% wide symmetric (σ 0.36, the poor eyes). Interocular-difference bands now fit: > 0.1 obs 285 / model 320, > 0.2 92 / 81, > 0.3 70 / 65, > 0.5 17 / 23.
- Sigmoid: γ 1.77 for good better eyes, falling to 0.78 by 0.5 logMAR (cutoff 0.26, τ 0.06). Binocular noise 0.044 rising to 0.145.
- Mean −0.058, SD 0.071 for the person value. Typical binocular −0.104 (SD 0.107 because the wide part is included); gain 0.046 over a single eye, 0.023 over the better eye.
- Binocular minus better eye, good better eye: diff 0 obs −0.022 / model −0.021; 0–0.1 −0.002 / −0.007; > 0.1 +0.007 / +0.015 (worse in 20% / 27%). So binocular loss for good eyes with a worse partner is already reproduced, slightly overstated; **no suppression term needed**. Poorer better eyes: the model still gives too little binocular benefit (diff 0–0.1 −0.084 / −0.056; > 0.1 −0.041 / −0.023).
- Unchanged misfits: binocular 20/20+ 164 vs 73; one eye bare 20/15 and other bare 20/20 125 vs 69; bare 20/20 + 20/20 minus 199 vs 319. Binocular misplaced 4.6%.
- To get mode B in every chain, `asym+sig+near` restricts the shift to 0.02–0.2 (s_w 0.01–0.1).

**`asym+sig+near` (current best; `BINOC_CHAINS=24 BINOC_TUNE=700 BINOC_DRAWS=500`, 109 min, trace `person_eyes2d_asym_sig_near.nc`):** 23 of 24 chains in mode B (log density −14189); chain 10 is stuck in mode C (−14210, `b2`/`b3` high). Dropping chain 10: R-hat ≤ 1.003, ESS ≥ 4,900 for the checked parameters, 0 divergences, log-likelihood −14116.6. All estimates match mode B above to the third decimal (shift 0.057 ± 0.036 in 26% of people; γ 1.76 → 0.79, cutoff 0.26). When using the trace, drop chains whose mean `logp` is about 20 below the rest (`idata.sel(chain=[...])`; see the scratchpad's `report_mode.py` pattern: `ext.report(idata.sel(chain=keep), full.load_triples(), "asym+sig+near")`).

Next things worth trying with this model: (1) the remaining binocular misfits (20/20+, 15/20 eye pairs, 20/20 + 20/20− eye pairs) may need recording style to vary within a visit, or the binocular letter windows to differ; (2) poorer better eyes still gain too little binocularly (model −0.055 vs obs −0.084 at diff 0–0.1, n = 21), a small group; (3) the notebook still uses `eyes_wide_nofilter` and should be pointed at this trace. (Done: the notebook now uses this trace.)

**Without large eye differences (asym+sig+near, simulated, not poor and no wide part, 89% of people):** monocular −0.050 / 0.077, better eye −0.064 / 0.073, binocular −0.088 / 0.078. Binocular advantage over the better eye 0.024 (0.022–0.026), over the mean of the eyes 0.038 (0.035–0.041). Near-identical eyes only (64%): 0.028 / 0.036.

**Applied without refitting to all ages:** per-visit log-likelihood 20–50 −3.92, 51–64 −4.30, < 20 −5.40, 65+ −6.05. Conditional log p(B | L, R) shortfall vs saturated 0.14, 0.29, 0.64, 0.52.

**Age bands, ages 20+ only (`fit_person_eyes2d_age.py`; ages < 20 are never fitted).** Bands 20–50 (3,604 visits), 51–64 (1,266), 65+ (1,184). Population parameters per band; recording rule, test noise, σ_d and the one-eye shift shared. 12 chains, tune 700, draws 500; both 0 divergences, R-hat ≤ 1.03 (worst: 65+ poor-vision parameters).
- `shared` (Ding & Sperling shared, 3.7 h): log-likelihood 20–50 −14140.1, 51–64 −5328.9, 65+ −6395.0 (untrained −5444.9, −7167.6). Conditional log p(B | L, R) −0.972 / −0.975 / −1.344 (saturated −0.836 / −0.706 / −0.889). With age: mean −0.058, −0.040, +0.023; SD about 0.075 in every band; wide-difference share 6%, 16%, 23%; one-eye-worse share 30%, 37%, 48% (at its 0.5 limit). Shared DS: γ 1.36 → 0.89, cutoff 0.14, σ_bin 0.046 → 0.130.
- `dsage` (ε, γ low/high, σ_bin low/high per band, 5.5 h): log-likelihood −14131.7, −5328.4, −6385.3, so +18.6 for 10 extra parameters; conditional −0.970 / −0.975 / −1.337, essentially unchanged. ε 4.3, 5.2, 6.6; γ_low about 1.6 in all bands; σ_bin_high 0.12, 0.19, 0.16. Age is in the eye distributions, not in how the eyes combine.
- Binocular minus better eye for equal eyes, better ≤ 20/25: obs −0.022 / −0.016 / −0.017, model about −0.020 in each band. Older bands lose overall gain only because more of them have unequal eyes. The model over-predicts binocular 20/15 at 65+ (180 vs 110).
- Unequal or poor eyes (scratchpad `bad_eyes.py`, pooled 20+): with the better eye ≤ 20/25 the mean B − better matches within 0.01 at every difference (0.1–0.3 +0.010 / +0.018, 0.3–0.6 +0.016 / +0.018), but the spread is too narrow for differences > 0.3 (SD 0.067 vs 0.053; > 0.6 0.096 vs 0.050, a few 65+ visits with B well below the better eye). With the better eye 20/30–20/40 the model is too noisy and too pessimistic: equal eyes SD 0.070 vs 0.106, B worse than better 2% vs 15%; difference 0.1–0.3 −0.029 vs −0.006. `dsage` does not fix this. The binocular record is rarely recorded worse than an equal poor better eye, which symmetric binocular noise cannot reproduce.
- Checks of `shared` (scratchpad `floor_agree.py`): (A) poor better eye (record 20/30–20/40), equal eyes, N = 79: B relative to better record, better 2+ 36 / 32, better 1 13 / 12, same 28 / 23, worse 1 2 / 9, worse 2+ 0 / 3, a one-sided floor. (B) Eye records identical: better ≤ 20/25 0.61 / 0.59 (N 5,791), 20/30–20/40 0.38 / 0.68 (N 208), worse than 20/40 0.29 / 0.68 (N 55): real poor eye pairs agree much less than the model, in every band. Each MRN appears once, so a noise term shared by L, R and B within a visit is the same as the person's true acuity and cannot be fitted.

**Noise variants (12 chains each, about 4 h, on the 20+ bands with `shared`):**
- `shared+dnoise` (binocular noise sqrt(σ_bin² + (k_diff·d)²)): k_diff 0.12; 1 divergence, R-hat ≤ 1.02. Log-likelihood −14145.7 / −5326.0 / −6366.5, total +26 over `shared`. Fixes the spread for very unequal good eyes (better ≤ 20/25, difference 0.3–0.6 SD obs 0.067 / model 0.066; > 0.6 0.096 / 0.093) but not the poor equal eyes (SD 0.104, B worse 14%). σ_bin_high only 0.130 → 0.124.
- `shared+tnoise` (eye-test noise σ_test + k_test·max(a − 0.1, 0)): **k_test 0.64**, so test noise about 0.07 at 0.2 and 0.14 at 0.3. 2 of 12 chains stuck (logp 65 lower, dropped); 0 divergences, R-hat ≤ 1.013. Log-likelihood −14086.9 / −5302.0 / −6283.4, **total +192 for one parameter**. Conditional 65+ −1.344 → −1.285. Eye agreement now fits (20/30–20/40 identical 0.38 / 0.39, 1 apart 0.31 / 0.32, 2+ 0.31 / 0.30). Poor equal eyes: mean −0.069 / −0.072, B worse 2% / 7%, floor table worse 1 2 / 3.5. So the floor is mostly test noise in the eye records, not a recording rule. Shifts DS: γ 1.95 → 0.39, cutoff 0.38, σ_bin 0.039 → 0.33. New misfits: unequal poor eyes get even less benefit (0.3–0.6 +0.079 vs −0.018, N 30), very unequal good eyes too wide (0.3–0.6 SD 0.119 vs 0.067), better eye worse than 20/40 (N 55) far too optimistic (−0.28 vs −0.06). The linear rise in test noise is probably too steep at the poor end.
- `shared+dnoise+tsig` (test noise on a sigmoid from σ_test to σ_test_high; 24 chains, 4.6 h): **not converged**. 3 chains dropped (logp 50–70 lower); of the rest R-hat up to 1.58 (cutoff, t_mid, sd_logmar, 65+ poor-vision part), ESS about 36; 0 divergences. The test noise is a step: σ_test 0.010 (at its lower limit) up to t_mid 0.16, then σ_test_high 0.154; t_tau 0.021 (at its 0.02 limit). k_diff falls to 0.02, so dnoise is no longer needed. DS back near `shared`: γ 1.82 → 0.48, cutoff 0.20, σ_bin 0.041 → 0.139. Log-likelihood −14083.9 / −5286.7 / −6294.9, total −25665.5, **only +6.8 over tnoise** (51–64 +15, 65+ −11.5). Conditional −0.972 / −0.956 / −1.283.
  - Better eye worse than 20/40 (N 43 in the table): obs −0.054, model −0.076 (tnoise −0.28), so this is fixed.
  - Very unequal good eyes, 0.3–0.6 (N 161): SD obs 0.067 / model 0.088 (tnoise 0.119, dnoise 0.066); mean +0.016 / +0.031. Better but still too wide.
  - Unequal poor eyes, better 20/30–20/40, difference 0.3–0.6 (N 30): obs −0.018, model +0.067 (tnoise +0.079). Still too little binocular benefit.
  - Poor equal eyes (N 111): SD 0.070 / 0.095, B worse 2% / 7.5%; floor worse 1 2 / 3.9. Agreement still fits (20/30–20/40 identical 0.38 / 0.37; worse than 20/40 0.29 / 0.20).
- `shared+tstep` (test noise σ_test below logMAR 0.15, σ_test_high above; 24 chains, 4 h): **best so far**, log-likelihood −14075.8 / −5283.2 / −6287.6, total −25646.6: +18.9 over tsig with 4 fewer parameters, +25.7 over tnoise with 1 more. Conditional −0.971 / −0.955 / −1.280. σ_test 0.010 (at its limit), σ_test_high 0.136. DS γ 1.81 → 0.50, cutoff 0.18, σ_bin 0.040 → 0.134. The poor-eye, floor and agreement tables match tsig to within 0.01. Still **not converged**, and not because of the noise: 6 of 24 chains about 93 lower (dropped); among the other 18 (logp within 4) R-hat up to 1.59 on the 65+ mean, SD and poor-vision part (w_poor 0.006–0.235, gap 0.11–0.96) and on cutoff and tau. With poor eyes now explained by test noise, the 65+ poor-vision part is weakly identified. tnoise (12 chains) had R-hat ≤ 1.013.
- `shared+tstep+poorshared` (as tstep, but the poor-vision gap and SD shared by all bands; only its share per band; 24 chains, 4.2 h, one chain stuck for the first 90 min then recovered): **converged and the best fit**. 5 chains dropped (logp 12–145 lower); the other 19 within 2, R-hat ≤ 1.010, ESS ≥ 2,100, 0 divergences. Log-likelihood −14076.8 / −5282.9 / −6273.9, total −25633.6: +13 over tstep with 4 fewer parameters. Conditional −0.971 / −0.954 / −1.280. Poor-vision part: gap 0.17, SD 0.39, share 3.6% / 4.1% / 5.2%. 65+ mean +0.031, SD 0.087. σ_test 0.0105 (at its limit), σ_test_high 0.137. DS: γ 1.75 → 0.55, cutoff 0.16, σ_bin 0.041 → 0.125, ε 5.0. Poor-eye, floor and agreement tables match tstep within 0.01, so the remaining misfits stand: very unequal good eyes (0.3–0.6, N 161) SD 0.067 / 0.086; unequal poor eyes (20/30–20/40, 0.3–0.6, N 30) −0.018 / +0.055; poor equal eyes (N 111) SD 0.070 / 0.097, B worse 2% / 8%.

**Progress output:** `sampling_progress.py` is used by `fit_person_records.py` and `fit_person_eyes2d.py`. Long runs print `[progress]` lines to stdout every `BINOC_REPORT_EVERY` seconds (default 60): tuning or sampling, percent done, slowest and fastest chain, divergences, minutes left. `BINOC_DRAWS` and `BINOC_TUNE` shorten a pilot run.

`record_types_by_reading` is out of date.

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
