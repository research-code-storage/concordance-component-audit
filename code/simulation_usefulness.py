#!/usr/bin/env python3
"""Summarize movement, cancellation and bound slack in the 72-scenario simulation.

The summaries include absolute net concordance change, mean absolute pairwise
movement, the tie-aware bound, replicate-level ratios and ratios of means.
Scenario means and Monte Carlo standard errors produce Tables S1 and S2.

Usage: python code/simulation_usefulness.py --study data/evidence
Outputs: generated/simulation_usefulness.json, table6.tex and si_simulation_tables.tex.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "generated")
    args = ap.parse_args()

    path = (args.study / "07_reference_study/results/simulation"
            / "all_14400_replicates.csv")
    rows = list(csv.DictReader(open(path)))
    if len(rows) != 14400:
        raise SystemExit(f"expected 14400 replicates, found {len(rows)}")

    keys = {(r["n"], r["strength"], r["component"], r["distribution"]) for r in rows}
    if len(keys) != 72:
        raise SystemExit(f"expected 72 scenarios, found {len(keys)}")

    expected={(str(n), str(a), c, d) for n in (50,200)
              for a in (0.01,0.1,0.5,1.0) for c in ('aligned','independent','opposed')
              for d in ('normal','t3','rounded_normal')}
    counts=Counter((r['n'],str(float(r['strength'])),r['component'],r['distribution']) for r in rows)
    if set(counts)!=expected or set(counts.values())!={200}:
        raise ValueError('Incomplete or duplicated simulation grid')
    identity=[(r['n'],r['strength'],r['component'],r['distribution'],r['replicate']) for r in rows]
    if len(set(identity))!=14400: raise ValueError('Duplicate simulation replicate')
    # the inequality itself: it must never be violated
    violations = sum(
        1 for r in rows
        if not (abs(float(r["signed_change"])) <= float(r["absolute_pair_change"]) + 1e-12
                <= float(r["tie_aware_bound"]) + 2e-12))

    if violations: raise ValueError(f"Bound violations: {violations}")

    def agg(rs):
        net = np.abs([float(r["signed_change"]) for r in rs])
        moved = np.array([float(r["absolute_pair_change"]) for r in rs])
        bound = np.array([float(r["tie_aware_bound"]) for r in rs])
        ok = net > 0
        se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v)))
        return {
            "replicates": len(rs),
            "net_zero": int((net==0).sum()),
            "moved_zero": int((moved==0).sum()),
            "complete_cancellations": int(((net==0)&(moved>0)).sum()),
            "hidden_denominator": int(ok.sum()),
            "slack_denominator": int((moved>0).sum()),
            "ratio_of_means": float(moved.mean()/net.mean()),
            "net": float(net.mean()),
            "moved": float(moved.mean()),
            "bound": float(bound.mean()),
            "hidden": float((moved[ok] / net[ok]).mean()) if ok.any() else float("nan"),
            "slack": float((bound[moved > 0] / moved[moved > 0]).mean()),
            "net_mcse": se(net), "moved_mcse": se(moved), "bound_mcse": se(bound),
        }

    out = {"violations": violations, "scenarios": len(keys), "replicates": len(rows),
           "by": {}}
    for factor in ("component", "distribution", "strength", "n"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[factor]].append(r)
        out["by"][factor] = {k: agg(v) for k, v in sorted(groups.items())}

    # every scenario, in the published order (n, strength, geometry, distribution)
    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["n"]), float(r["strength"]), r["component"], r["distribution"])].append(r)
    order = [(n, a, c, d) for n in (50, 200) for a in (0.01, 0.1, 0.5, 1.0)
             for c in ("aligned", "independent", "opposed")
             for d in ("normal", "t3", "rounded_normal")]
    out["scenario_cells"] = [{"n": n, "strength": a, "component": c, "distribution": d,
                         **agg(cells[(n, a, c, d)])} for n, a, c, d in order]
    # limits of Proposition 3 at small strengths (ratio of mean bound to mean movement)
    out["small_strength_ratio"] = {}
    for dist in ("normal", "t3", "rounded_normal"):
        sel = [r for r in rows if float(r["strength"]) <= 0.1 and r["distribution"] == dist]
        mv = np.array([float(r["absolute_pair_change"]) for r in sel])
        bd = np.array([float(r["tie_aware_bound"]) for r in sel])
        out["small_strength_ratio"][dist] = {
            "by_strength": {str(a): float(bd[[float(r["strength"]) == a for r in sel]].mean()
                                          / mv[[float(r["strength"]) == a for r in sel]].mean())
                            for a in (0.01, 0.1)},
            "bound_equals_movement_every_replicate": bool(np.all(bd == mv))}
    comp = out["by"]["component"]
    out["max_component_mcse"] = {k: max(v[k + "_mcse"] for v in comp.values())
                                 for k in ("net", "moved", "bound")}

    print(f"{len(rows)} replicates over {len(keys)} scenarios; "
          f"bound violations: {violations}")
    for factor, table in out["by"].items():
        print(f"\nby {factor}")
        print(f"  {'level':16s}{'net':>9}{'moved':>9}{'bound':>9}{'hidden':>9}{'slack':>9}")
        for k, v in table.items():
            print(f"  {k:16s}{v['net']:9.4f}{v['moved']:9.4f}{v['bound']:9.4f}"
                  f"{v['hidden']:9.2f}{v['slack']:9.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "simulation_usefulness.json").write_text(json.dumps(out, indent=1))

    # main-text table: the component factor, which is the one that separates the regimes
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Net concordance change, movement and bound across component geometries. Each row averages 4800 "
         r"replicates over the 24 scenarios sharing that component geometry, from a "
         r"$2\times4\times3\times3$ design over sample size, signal strength, component "
         r"geometry and score distribution. \emph{Net} is the absolute change in "
         r"concordance after deleting the component, \emph{moved} the mean absolute "
         r"pairwise movement, \emph{bound} the tie-aware bound of "
         r"Proposition~1. \emph{Hidden} is the mean replicate-level moved/net ratio "
         r"conditional on net$>0$ (4694, 4612 and 4689 replicates, respectively); "
         r"\emph{slack} is the mean bound/moved ratio conditional on moved$>0$ "
         r"(4751, 4741 and 4757 replicates). These are not ratios of the displayed "
         r"column means. Monte Carlo standard errors of the displayed means are at most "
         rf"{out['max_component_mcse']['net']:.4f} (net), {out['max_component_mcse']['moved']:.4f} "
         rf"(moved) and {out['max_component_mcse']['bound']:.4f} (bound). The bound was violated "
         r"in none of the 14400 replicates; Supporting Information~E reports every scenario.}",
         r"\label{tab:usefulness}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Component geometry & Net & Moved & Bound & Hidden & Slack \\", r"\midrule"]
    names = {"aligned": "Aligned", "independent": "Independent", "opposed": "Opposed"}
    for k, v in out["by"]["component"].items():
        L.append(f"{names.get(k, k)} & {v['net']:.4f} & {v['moved']:.4f} & "
                 f"{v['bound']:.4f} & {v['hidden']:.2f} & {v['slack']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.out / "table6.tex").write_text("\n".join(L) + "\n")
    print(f"\nwrote {args.out/'table6.tex'}")
    (args.out / "si_simulation_tables.tex").write_text(si_tables(out["scenario_cells"]))
    print(f"wrote {args.out/'si_simulation_tables.tex'}")


def si_tables(scenarios):
    """Supporting Information Tables S1 (n=50) and S2 (n=200): all 72 scenarios."""
    geo = {"aligned": "Aligned", "independent": "Independent", "opposed": "Opposed"}
    dist = {"normal": "Normal", "t3": "$t_3$", "rounded_normal": "Rounded"}

    def f(m, se):
        return f"{m:.4f} ({se:.4f})" if se >= 5e-5 else f"{m:.4f} ($<$0.0001)"

    blocks = []
    for num, n in ((1, 50), (2, 200)):
        L = [rf"\noindent\textbf{{Table S{num}:}} Per-scenario results of the 72-scenario "
             rf"simulation, $n={n}$ ({n // 2} cases, {n // 2} controls, {n * n // 4} pairs). "
             r"Entries are means over 200 replicates with Monte Carlo standard errors in "
             r"parentheses. Net is $|\Delta|$, moved is $A_\omega$ and bound is the tie-aware "
             r"$B_\omega$.",
             r"\begin{center}\footnotesize\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lllccc}",
             r"\toprule", r"Strength $a$ & Geometry & Distribution & Net & Moved & Bound \\",
             r"\midrule"]
        previous = None
        for c in (c for c in scenarios if c["n"] == n):
            if previous is not None and c["strength"] != previous:
                L.append(r"\addlinespace")
            previous = c["strength"]
            assert c["replicates"] == 200
            L.append(f"{c['strength']:g} & {geo[c['component']]} & {dist[c['distribution']]} & "
                     f"{f(c['net'], c['net_mcse'])} & {f(c['moved'], c['moved_mcse'])} & "
                     f"{f(c['bound'], c['bound_mcse'])} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
        blocks.append("\n".join(L))
    return "\n\\clearpage\n".join(blocks) + "\n"


if __name__ == "__main__":
    main()
