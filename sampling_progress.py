"""Progress reporting for long PyMC fits sampled with nutpie.

sample() compiles the model with nutpie, then samples in the background and prints a status
line every BINOC_REPORT_EVERY seconds (default 60): draws done per chain, divergences so far,
and an estimated time left from the rate so far.
BINOC_DRAWS and BINOC_TUNE override the draw counts, for a short pilot run.
All lines start with [progress] and are flushed, so they show up in a log file as they happen.
"""

import os
import time

import numpy as np
import nutpie


def say(text):
    print(f"[progress {time.strftime('%H:%M:%S')}] {text}", flush=True)


def finite_draws(ds, name):
    """Draws filled in so far for each chain (unfilled draws are NaN)."""
    if ds is None or name not in ds:
        return None
    values = ds[name].values
    return np.isfinite(values).reshape(values.shape[0], -1).all(axis=1) if values.ndim > 2 else np.isfinite(values)


def status(sampler):
    """Draws done per chain, and divergences in tuning and in sampling so far."""
    trace = sampler.inspect()
    parts = []
    divergences = []
    for group in ["warmup_sample_stats", "sample_stats"]:
        ds = getattr(trace, group, None)
        filled = finite_draws(ds, "step_size")
        if filled is None:
            divergences.append(0)
            continue
        parts.append(filled.sum(axis=1))
        divergences.append(int(np.nansum(np.where(filled, ds["diverging"].values, 0))) if "diverging" in ds else 0)
    if not parts:
        return None, divergences
    return np.sum(parts, axis=0), divergences


def sample(model, chains, draws, tune, target_accept=0.8, random_seed=1, **_):
    draws = int(os.environ.get("BINOC_DRAWS", draws))
    tune = int(os.environ.get("BINOC_TUNE", tune))
    every = float(os.environ.get("BINOC_REPORT_EVERY", 60))
    say("compiling the model")
    start = time.time()
    compiled = nutpie.compile_pymc_model(model)
    say(f"compiled in {time.time() - start:.0f} s; {chains} chains, {tune} tune + {draws} draws each")
    start = time.time()
    sampler = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, cores=chains, seed=random_seed,
                            target_accept=target_accept, blocking=False, progress_bar=False)
    total = tune + draws
    while not sampler.is_finished:
        next_report = time.time() + every
        while time.time() < next_report and not sampler.is_finished:
            time.sleep(1)
        if sampler.is_finished:
            break
        try:
            counts, divergences = status(sampler)
        except Exception as err:
            say(f"running {(time.time() - start) / 60:.1f} min (status unavailable: {err})")
            continue
        if counts is None:
            say(f"running {(time.time() - start) / 60:.1f} min, no draws yet")
            continue
        elapsed = time.time() - start
        share = counts.sum() / (total * chains)
        eta = elapsed / share - elapsed if share > 0 else float("nan")
        phase = "tuning" if counts.min() < tune else "sampling"
        say(f"{elapsed / 60:.1f} min, {phase}, {100 * share:.0f}% done "
            f"(slowest chain {int(counts.min())}/{total}, fastest {int(counts.max())}/{total}), "
            f"divergences {divergences[0]} in tuning, {divergences[1]} in sampling, about {eta / 60:.1f} min left")
    idata = sampler.wait()
    say(f"sampling finished in {(time.time() - start) / 60:.1f} min")
    return idata
