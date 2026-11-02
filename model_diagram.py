"""Render the learned-cutoff model as a graphical model (plate diagram) with PyMC.

Writes learned_cutoff_model.png (and .pdf) in this folder. Needs graphviz (system `dot`
and the python `graphviz` package).
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

HERE = Path(__file__).parent


def build_graph_model():
    # Real row count, so the plate shows N; values are placeholders (graph is value-independent).
    from fit_ding_sperling_decimal_pymc import load_all_rows
    from fit_ding_sperling_discrete_pymc import bin_edges
    kb, kw, y = load_all_rows()
    lo, up = bin_edges(y)
    x = -np.log10(kb)
    with pm.Model() as model:
        log_eps = pm.Normal("log_eps", mu=0.0, sigma=2.0)
        eps = pm.Deterministic("eps", pt.exp(log_eps))
        g_lo = pm.LogNormal("gamma_low", mu=np.log(2.5), sigma=1.0)
        g_hi = pm.LogNormal("gamma_high", mu=np.log(0.7), sigma=1.0)
        s_lo = pm.HalfNormal("sigma_low", sigma=0.1)
        s_hi = pm.HalfNormal("sigma_high", sigma=0.3)
        c = pm.Uniform("cutoff", lower=0.1, upper=1.0)
        tau = pm.Deterministic("tau", 0.005 + pm.HalfNormal("tau_raw", sigma=0.05))

        w = pm.Deterministic("w", pm.math.sigmoid((x - c) / tau))
        g = pm.Deterministic("gamma_i", g_lo + (g_hi - g_lo) * w)
        sig = pm.Deterministic("sigma_i", s_lo + (s_hi - s_lo) * w)
        ab = eps * kb**g
        aw = eps * kw**g
        k_ou = ((1.0 + ab) * kb + (1.0 + aw) * kw) / (1.0 + ab + aw)
        mu = pm.Deterministic("mu_logmar", -pt.log10(k_ou))

        def cdf(v):
            return 0.5 * (1.0 + pt.erf((v - mu) / (sig * np.sqrt(2.0))))

        prob = cdf(up) - cdf(lo)
        pm.Potential("bin_loglik", pt.sum(pt.log(pt.maximum(prob, 1e-12))))
    return model


def main():
    model = build_graph_model()
    g = pm.model_to_graphviz(model)
    g.graph_attr.update(rankdir="LR", fontname="Helvetica")
    g.node_attr.update(fontname="Helvetica")
    out_base = HERE / "learned_cutoff_model"
    g.render(str(out_base), format="png", cleanup=True)
    g.render(str(out_base), format="pdf", cleanup=True)
    print(f"Saved -> {out_base}.png and {out_base}.pdf")


if __name__ == "__main__":
    main()
