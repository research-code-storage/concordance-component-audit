#!/usr/bin/env python3
"""Closed-form variance of Proposition 4 for the lung-cancer component analysis.

The fitted Cox coefficients are held fixed and the 69 held-out patients are treated as an
independent sample.  For a bounded pair kernel kappa on the comparable pairs (i, j), with
status_i = 1 and time_i < time_j and unit weights, the statistic T = sum(kappa) / M is a
ratio of two U-statistics, and Proposition 4 gives the variance estimator

    V = N / {(N - 1) M^2} * sum_i {S_i(kappa) - T * S_i(1)}^2,

where S_i(kappa) is the sum of kappa over the comparable pairs that contain patient i and
S_i(1) is their number.  Kernels: Delta (change in pair contribution), A (its absolute
value), B (tie-aware bound rho) and V (inclusive screen), for both same-fit deletions and
for the paired differences between them.  Normal 95% intervals use z = Phi^{-1}(0.975).

The estimator is computed twice, from the pair sums above and from an explicit Hoeffding
projection of the symmetrized numerator and denominator kernels, and the two must agree.
On a binary outcome (event versus no event) the estimator must reproduce the
structural-component display of Supporting Information B exactly.  A leave-one-patient-out
jackknife variance is reported as a further check, and the standard errors are compared
with those of the patient bootstrap (code/lung_bootstrap.py, which must run first).  No
resampling and no random numbers are used here.

Usage:  python code/lung_ustat.py --study data/evidence
Writes: generated/lung_ustat.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

KEYS = ("Delta", "A", "B", "V")
# deleted component -> (retained column, deleted column)
DELETIONS = {
    "delete_age_sex": ("ecog_component", "age_sex_component"),
    "delete_ecog": ("age_sex_component", "ecog_component"),
}
Z975 = float(norm.ppf(0.975))


def H(x):
    return (x > 0) + 0.5 * (x == 0)


def rho(D, E):
    aD, aE = np.abs(D), np.abs(E)
    r = np.zeros(D.shape)
    nz = D != 0
    r[nz & (aE > aD)] = 1.0
    r[nz & (aE == aD)] = 0.5
    r[(~nz) & (E != 0)] = 0.5
    return r


def kernels(data, retained, deleted):
    """Comparability matrix c[i, j] and the four kernels on ordered pairs (i, j)."""
    t, s = data["time"], data["status"]
    c = ((s[:, None] == 1) & (t[:, None] < t[None, :])).astype(float)
    r, d = data[retained], data[deleted]
    D = r[:, None] - r[None, :]
    E = d[:, None] - d[None, :]
    delta = H(D + E) - H(D)
    return c, {"Delta": delta, "A": np.abs(delta), "B": rho(D, E),
               "V": (np.abs(D) <= np.abs(E)).astype(float)}


def closed_form(c, k):
    """Statistic and Proposition 4 variance from the per-patient pair sums."""
    N = c.shape[0]
    M = c.sum()
    ck = c * k
    T = ck.sum() / M
    S_k = ck.sum(1) + ck.sum(0)   # pairs containing i as first or as second member
    S_1 = c.sum(1) + c.sum(0)
    return T, N / ((N - 1) * M ** 2) * np.sum((S_k - T * S_1) ** 2)


def projection(c, k):
    """The same estimator from the Hoeffding projections of the symmetric kernels."""
    N = c.shape[0]
    h1 = 0.5 * (c * k + (c * k).T)
    h0 = 0.5 * (c + c.T)
    np.fill_diagonal(h1, 0.0)
    np.fill_diagonal(h0, 0.0)
    U1, U0 = h1.sum() / (N * (N - 1)), h0.sum() / (N * (N - 1))
    T = U1 / U0
    psi = (h1.sum(1) / (N - 1) - T * h0.sum(1) / (N - 1)) / U0
    return T, 4.0 * np.sum(psi ** 2) / (N * (N - 1))


def jackknife(c, k):
    """Leave-one-patient-out jackknife variance of the same ratio statistic."""
    N = c.shape[0]
    ck = c * k
    S_k = ck.sum(1) + ck.sum(0)
    S_1 = c.sum(1) + c.sum(0)
    loo = (ck.sum() - S_k) / (c.sum() - S_1)
    return (N - 1) / N * np.sum((loo - loo.mean()) ** 2)


def binary_reduction(data):
    """Binary outcome with unit weights on case-control pairs (event versus no event, full
    score): the Proposition 4 estimator must equal the display of Supporting Information B,
    N/(N-1) {sum_i (V10_i - T)^2 / m^2 + sum_j (V01_j - T)^2 / n^2}."""
    y, f = data["status"], data["full_score"]
    c = ((y[:, None] == 1) & (y[None, :] == 0)).astype(float)
    k = H(f[:, None] - f[None, :])
    T, var = closed_form(c, k)
    cases, controls = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n, N = len(cases), len(controls), len(y)
    block = k[np.ix_(cases, controls)]
    V10, V01 = block.mean(1), block.mean(0)
    display = N / (N - 1) * (np.sum((V10 - T) ** 2) / m ** 2 + np.sum((V01 - T) ** 2) / n ** 2)
    assert abs(var - display) <= 1e-12 * var, (var, display)
    delong = np.sum((V10 - T) ** 2) / (m * (m - 1)) + np.sum((V01 - T) ** 2) / (n * (n - 1))
    return {"status": "PASS", "cases": m, "controls": n, "statistic": float(T),
            "proposition_6_variance": float(var), "structural_component_variance": float(delong)}


def summarise(c, k, boot_se):
    T, var = closed_form(c, k)
    T2, var2 = projection(c, k)
    assert abs(T - T2) < 1e-12 and abs(var - var2) <= 1e-12 * max(var, 1e-300), (T, T2, var, var2)
    se = float(np.sqrt(var))
    return {"estimate": float(T), "se": se, "lower": float(T - Z975 * se),
            "upper": float(T + Z975 * se), "jackknife_se": float(np.sqrt(jackknife(c, k))),
            "bootstrap_se": float(boot_se), "se_over_bootstrap_se": float(se / boot_se)}


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
    boot = json.loads((a.out / "lung_bootstrap.json").read_text())

    kern = {}
    out = {"method": "Proposition 4: ratio of U-statistics, Hoeffding projection and delta method",
           "variance": "N/((N-1)M^2) * sum_i (S_i(kappa) - T S_i(1))^2",
           "interval": "estimate +/- z * se, z = Phi^{-1}(0.975)", "z": Z975,
           "coefficients": "held fixed at the training fit", "n_patients": n}
    for key, (ret, dele) in DELETIONS.items():
        c, kern[key] = kernels(data, ret, dele)
        out["n_comparable_pairs"] = int(c.sum())
        out[key] = {}
        for q in KEYS:
            o = summarise(c, kern[key][q], boot[key][q]["bootstrap_se"])
            assert abs(o["estimate"] - boot[key][q]["estimate"]) < 1e-12, (key, q)
            out[key][q] = o
    assert out["n_comparable_pairs"] == 1653
    for q in ("Delta", "A"):
        name = f"paired_difference_{q}_age_sex_minus_ecog"
        o = summarise(c, kern["delete_age_sex"][q] - kern["delete_ecog"][q], boot[name]["bootstrap_se"])
        assert abs(o["estimate"] - boot[name]["estimate"]) < 1e-12, name
        out[name] = o
    ratios = [out[k][q]["se_over_bootstrap_se"] for k in DELETIONS for q in KEYS] + \
             [out[f"paired_difference_{q}_age_sex_minus_ecog"]["se_over_bootstrap_se"] for q in ("Delta", "A")]
    out["se_over_bootstrap_se_range"] = [float(min(ratios)), float(max(ratios))]
    out["binary_reduction_check"] = binary_reduction(data)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "lung_ustat.json").write_text(json.dumps(out, indent=1))
    for k in list(DELETIONS) + [f"paired_difference_{q}_age_sex_minus_ecog" for q in ("Delta", "A")]:
        items = out[k].items() if k in DELETIONS else [("", out[k])]
        for q, o in items:
            print(f"{k:42s} {q:5s} {o['estimate']:+.4f}  se {o['se']:.4f}  [{o['lower']:.3f}, {o['upper']:.3f}]"
                  f"  jackknife se {o['jackknife_se']:.4f}  bootstrap se {o['bootstrap_se']:.4f}")
    print("closed-form / bootstrap standard error: {:.2f} to {:.2f}".format(*out["se_over_bootstrap_se_range"]))


if __name__ == "__main__":
    main()
