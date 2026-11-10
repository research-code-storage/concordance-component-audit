#!/usr/bin/env python3
"""Recompute application point estimates from saved component scores.

Profile scores are averaged within compound, then used to calculate concordance,
the tie-aware bound of Proposition 1 and inclusive-screen pair counts. Results
are compared with the supplied numerical records.

Usage: python code/compute_all.py --release data/evidence
Outputs: generated/results.json, generated/crosscheck.txt and table inputs.
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# configuration                                                               #
# --------------------------------------------------------------------------- #

PANELS = {
    "PHH": "PHH",
    "external": "DILImap",
    "source": "Within-source holdout",
}

DESIGNS = [
    ("historical_centred", "Pooled, centred"),
    ("matched_centred", "Matched, centred"),
    ("revised_origin", "Matched, origin preserving"),
]

READOUTS = [
    ("softplus_original", "Full predictor"),
    ("fixed_sum", "Fixed sum"),
    ("kernel_original", "Kernel only"),
]

MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = range(10)
SIZES = (60,)


# --------------------------------------------------------------------------- #
# concordance and the Proposition 1 bound                                     #
# --------------------------------------------------------------------------- #

def concordance(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware AUC: H(x) = 1{x>0} + 0.5*1{x=0} averaged over all cross-class pairs."""
    d = pos[:, None] - neg[None, :]
    return float((np.sign(d) > 0).mean() + 0.5 * (d == 0).mean())


def pair_budget(d_pos, d_neg, e_pos, e_neg, t: float = 1.0):
    """Proposition 1 bounds for deleting component ``e`` from ``d + t*e``.

    D is the retained (linear) difference, E the deleted (kernel) difference.
    Returns (tie-aware bound B, inclusive flagged count V, n_pairs).

    Per-pair maximum absolute contribution change rho(D, E):
        D != 0, |E| <  |D|  -> 0     (sign of D+E cannot differ from sign of D)
        D != 0, |E| == |D|  -> 1/2   (adverse sign creates a tie)
        D != 0, |E| >  |D|  -> 1     (adverse sign reverses the pair)
        D == 0, E  != 0     -> 1/2   (leaving a tie)
        D == 0, E  == 0     -> 0
    """
    D = d_pos[:, None] - d_neg[None, :]
    E = t * (e_pos[:, None] - e_neg[None, :])
    aD, aE = np.abs(D), np.abs(E)

    rho = np.zeros_like(D, dtype=float)
    nz = D != 0
    rho[nz & (aE > aD)] = 1.0
    rho[nz & (aE == aD)] = 0.5
    z = ~nz
    rho[z & (E != 0)] = 0.5

    vulnerable = int((aD <= aE).sum())
    return float(rho.mean()), vulnerable, int(D.size)


def actual_change(d_pos, d_neg, e_pos, e_neg, t: float = 1.0):
    """Signed AUC change and the number of pairs whose contribution actually changed."""
    D = d_pos[:, None] - d_neg[None, :]
    E = t * (e_pos[:, None] - e_neg[None, :])
    h_full = (np.sign(D + E) > 0) + 0.5 * ((D + E) == 0)
    h_lin = (np.sign(D) > 0) + 0.5 * (D == 0)
    return float(h_full.mean() - h_lin.mean()), int((h_full != h_lin).sum())


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #

def load_panel(exp: Path, n: int, seed: int, domain: str):
    """Load one saved score file and aggregate profile scores within compound."""
    z = np.load(exp / f"n{n}_seed{seed}_{domain}_scores.npz", allow_pickle=True)
    arms = [str(a) for a in z["arm_names"]]
    compound = np.asarray([str(c) for c in z["compound"]])
    label = np.asarray(z["label"]).astype(int)

    order = sorted(set(compound))
    idx = {c: compound == c for c in order}
    y = np.array([int(label[idx[c]][0]) for c in order])
    if not all(len(set(label[idx[c]].tolist())) == 1 for c in order):
        raise ValueError(f"within-compound label conflict in n{n}_seed{seed}_{domain}")

    def agg(mat):
        return np.stack([mat[idx[c]].mean(axis=0) for c in order])

    return {
        "arms": arms,
        "compounds": order,
        "y": y,
        "full": agg(z["full"]),
        "linear": agg(z["linear"]),
        "kernel": agg(z["kernel"]),
    }


def fixed_sum_scores(exp: Path, n: int, seed: int, domain: str, design: str):
    """The unfitted two-coordinate sum, stored as the design's diagnostic column."""
    z = np.load(exp / f"n{n}_seed{seed}_{domain}_scores.npz", allow_pickle=True)
    names = [str(s) for s in z["diagnostic_names"]]
    col = names.index(design)
    compound = np.asarray([str(c) for c in z["compound"]])
    order = sorted(set(compound))
    return np.array([z["diagnostic"][compound == c, col].mean() for c in order])


# --------------------------------------------------------------------------- #
# main computation                                                             #
# --------------------------------------------------------------------------- #

def compute(release: Path):
    exp = release / "05_reconstruction/results/experiments"
    out = {}

    for n in SIZES:
        for domain in PANELS:
            per_seed = [load_panel(exp, n, s, domain) for s in SEEDS]
            arms = per_seed[0]["arms"]
            y = per_seed[0]["y"]
            pos, neg = y == 1, y == 0

            for design, _ in DESIGNS:
                # ---- fitted prediction scores -------------------------------------- #
                for readout, _ in READOUTS:
                    if readout == "fixed_sum":
                        continue
                    arm = f"{design}__{readout}"
                    if arm not in arms:
                        continue
                    a = arms.index(arm)

                    aucs, budgets, vulns, changes, changed_pairs = [], [], [], [], []
                    curves = {f"{t:g}": [] for t in MULTIPLIERS}
                    for p in per_seed:
                        full, lin, ker = p["full"][:, a], p["linear"][:, a], p["kernel"][:, a]
                        aucs.append(concordance(full[pos], full[neg]))
                        b, v, npairs = pair_budget(lin[pos], lin[neg], ker[pos], ker[neg])
                        budgets.append(b)
                        vulns.append(v)
                        d, c = actual_change(lin[pos], lin[neg], ker[pos], ker[neg])
                        changes.append(d)
                        changed_pairs.append(c)
                        for t in MULTIPLIERS:
                            curves[f"{t:g}"].append(
                                pair_budget(lin[pos], lin[neg], ker[pos], ker[neg], t)[0]
                            )

                    out[f"{n}|{domain}|{arm}"] = {
                        "auc_mean": float(np.mean(aucs)),
                        "auc_per_draw": aucs,
                        "budget_per_draw": budgets,
                        "budget_max": float(max(budgets)),
                        "vulnerable_per_draw": vulns,
                        "vulnerable_min": int(min(vulns)),
                        "vulnerable_max": int(max(vulns)),
                        "actual_change_per_draw": changes,
                        "changed_pairs_per_draw": changed_pairs,
                        "changed_pairs_min": int(min(changed_pairs)),
                        "changed_pairs_max": int(max(changed_pairs)),
                        "n_pairs": npairs,
                        "budget_curve": {k: float(np.max(v)) for k, v in curves.items()},
                        "budget_curve_per_draw": {k: [float(x) for x in v]
                                                  for k, v in curves.items()},
                        "n_compounds": len(y),
                        "n_positive": int(pos.sum()),
                        "n_negative": int(neg.sum()),
                    }

                # ---- the unfitted fixed sum ------------------------------- #
                fs = [fixed_sum_scores(exp, n, s, domain, design) for s in SEEDS]
                out[f"{n}|{domain}|{design}__fixed_sum"] = {
                    "auc_mean": float(np.mean([concordance(f[pos], f[neg]) for f in fs])),
                    "auc_per_draw": [concordance(f[pos], f[neg]) for f in fs],
                    "n_compounds": len(y),
                    "n_positive": int(pos.sum()),
                    "n_negative": int(neg.sum()),
                    "note": "no fitted kernel; deletion bound not defined",
                }
    return out


def crosscheck(out: dict, release: Path):
    """Compare recomputed AUCs with the saved summary table."""
    rows = list(csv.DictReader(
        open(release / "05_reconstruction/results/experiments/all_results.csv")))
    rows = [r for r in rows if r["domain"] in PANELS]  # reported evaluation panels only
    lines, worst = [], 0.0
    for r in rows:
        key = f"{r['training_compounds']}|{r['domain']}|{r['arm']}"
        if key not in out:
            lines.append(f"MISSING from recomputation: {key}")
            continue
        a, b = float(r["auc_mean"]), out[key]["auc_mean"]
        worst = max(worst, abs(a - b))
        if abs(a - b) > 1e-9:
            lines.append(f"DISAGREE {key}: saved {a!r} recomputed {b!r}")
    lines.append(f"maximum |saved - recomputed| AUC difference: {worst:.3e}")
    lines.append(f"compared {len(rows)} saved summary cells")
    return "\n".join(lines), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "generated")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    out = compute(args.release)
    (args.out / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True))

    report, worst = crosscheck(out, args.release)
    (args.out / "crosscheck.txt").write_text(report + "\n")
    print(report)
    if worst > 1e-9:
        raise SystemExit("recomputation disagrees with the saved summary")
    print(f"\nwrote {args.out/'results.json'}")


if __name__ == "__main__":
    main()
