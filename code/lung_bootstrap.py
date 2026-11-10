#!/usr/bin/env python3
"""Patient-level bootstrap intervals for the lung-cancer component analysis.

The fitted Cox coefficients are held fixed. Each replicate resamples the 69 held-out
patients with replacement, re-forms the comparable pairs among the resampled patients
(a patient paired with its own copy is never comparable, because the times are equal),
and recomputes, for both same-fit deletions,

  Delta  net concordance change  C(full) - C(retained)
  A      mean absolute per-pair contribution change
  B      tie-aware bound of Proposition 1
  V      inclusive screen (share of pairs with |D| <= |E|)

The intervals describe evaluation-sample variability only; they do not include
uncertainty in the fitted coefficients. Percentile intervals, 5000 replicates,
seed 20261005.

Usage:  python code/lung_bootstrap.py --study data/evidence
Writes: generated/lung_bootstrap.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPLICATES = 5000
SEED = 20261005
KEYS = ("Delta", "A", "B", "V")
# deleted component -> (retained column, deleted column)
DELETIONS = {
    "delete_age_sex": ("ecog_component", "age_sex_component"),
    "delete_ecog": ("age_sex_component", "ecog_component"),
}


def H(x):
    return (x > 0) + 0.5 * (x == 0)


def rho(D, E):
    aD, aE = np.abs(D), np.abs(E)
    r = np.zeros(len(D))
    nz = D != 0
    r[nz & (aE > aD)] = 1.0
    r[nz & (aE == aD)] = 0.5
    r[(~nz) & (E != 0)] = 0.5
    return r


def pairs(t, s):
    n = len(t)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ok = (s[I] == 1) & (t[I] < t[J])
    return I[ok], J[ok]


def audit(data, idx, retained, deleted):
    t, s = data["time"][idx], data["status"][idx]
    i, j = pairs(t, s)
    r, d = data[retained][idx], data[deleted][idx]
    D, E = r[i] - r[j], d[i] - d[j]
    hf, hr = H(D + E), H(D)
    return {"pairs": int(len(i)),
            "Delta": float(hf.mean() - hr.mean()),
            "A": float(np.abs(hf - hr).mean()),
            "B": float(rho(D, E).mean()),
            "V": float((np.abs(D) <= np.abs(E)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "generated")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.study / "07_reference_study/results/survival/heldout_scores.csv")))
    data = {k: np.array([float(r[k]) for r in rows]) for k in
            ("time", "status", "age_sex_component", "ecog_component", "full_score")}
    data["status"] = data["status"].astype(int)
    n = len(rows)
    assert n == 69 and int(data["status"].sum()) == 47
    np.testing.assert_allclose(data["age_sex_component"] + data["ecog_component"],
                               data["full_score"], rtol=0, atol=1e-12)

    point = {k: audit(data, np.arange(n), *v) for k, v in DELETIONS.items()}
    assert point["delete_age_sex"]["pairs"] == point["delete_ecog"]["pairs"] == 1653

    rng = np.random.default_rng(SEED)
    boot = {k: {q: [] for q in KEYS} for k in DELETIONS}
    for _ in range(REPLICATES):
        idx = rng.integers(0, n, n)
        for k, (ret, dele) in DELETIONS.items():
            r = audit(data, idx, ret, dele)
            for q in KEYS:
                boot[k][q].append(r[q])

    out = {"replicates": REPLICATES, "seed": SEED, "resampling_unit": "held-out patient",
           "interval": "percentile 2.5/97.5", "coefficients": "held fixed at the training fit",
           "n_patients": n, "n_comparable_pairs": point["delete_age_sex"]["pairs"]}
    for k in DELETIONS:
        out[k] = {}
        for q in KEYS:
            v = np.array(boot[k][q])
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[k][q] = {"estimate": point[k][q], "lower": float(lo), "upper": float(hi),
                         "bootstrap_se": float(v.std(ddof=1))}
    for q in ("Delta", "A"):
        v = np.array(boot["delete_age_sex"][q]) - np.array(boot["delete_ecog"][q])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"paired_difference_{q}_age_sex_minus_ecog"] = {
            "estimate": point["delete_age_sex"][q] - point["delete_ecog"][q],
            "lower": float(lo), "upper": float(hi), "bootstrap_se": float(v.std(ddof=1))}

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "lung_bootstrap.json").write_text(json.dumps(out, indent=1))
    for k in DELETIONS:
        print(k)
        for q in KEYS:
            o = out[k][q]
            print(f"  {q:6s} {o['estimate']:.4f}  [{o['lower']:.3f}, {o['upper']:.3f}]")
    for q in ("Delta", "A"):
        o = out[f"paired_difference_{q}_age_sex_minus_ecog"]
        print(f"paired {q} (age and sex minus ECOG): [{o['lower']:.3f}, {o['upper']:.3f}]")


if __name__ == "__main__":
    main()
