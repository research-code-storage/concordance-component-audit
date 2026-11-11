#!/usr/bin/env python3
"""Same-fit deletion versus refitting for the lung-cancer illustration.

The clinical analysis (Section 4.1 and Supplementary Information H) compares component
deletion from a fitted Cox model with reduced-model refitting on the same 158 training
patients.

1. The full three-covariate fit is replayed from the complete-case snapshot and must
   reproduce the saved coefficients and the saved held-out scores.
2. The two reduced Cox models (age and sex; ECOG alone) are refitted on the same training
   patients, with the same standardization, Efron ties and convergence check.
3. On the 69 held-out patients and their 1653 comparable pairs, each refitted reduced
   score is compared with the full score (net change and movement, with strict reversals
   and tie transitions) and with the same-fit retained score.  Closed-form intervals of
   Proposition 4 hold the fitted coefficients fixed, as in code/lung_ustat.py.
4. A training-sample bootstrap (1000 resamples of the 158 training patients, seed
   20261019) refits the full and both reduced models in every replicate and recomputes the
   same-fit analysis and the refitted contrasts on the fixed held-out sample.  These intervals
   describe model-fitting variability conditional on the held-out patients, which the
   evaluation-sample intervals of Section 4.1 exclude.

The bound B is reported only for the same-fit decomposition: a refitted reduced model
is not obtained by deleting a component of the full score, so Proposition 1 does not apply
to it with the deleted block as the component.

Usage:  python code/survival_refit.py --study data/evidence
Writes: generated/survival_refit.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from statsmodels.duration.hazard_regression import PHReg

from lung_ustat import Z975, closed_form, projection

REPLICATES = 1000
SEED = 20261019
FEATURES = ("age", "sex", "ph.ecog")
# deleted block -> (columns kept by the reduced model, retained and deleted held-out columns)
DELETIONS = {
    "delete_age_sex": ((2,), "ecog_component", "age_sex_component"),
    "delete_ecog": ((0, 1), "age_sex_component", "ecog_component"),
}


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


def cox(time, status, X):
    model = PHReg(time, X, status=status, ties="efron")
    fit = model.fit(disp=False, maxiter=1000)
    gradient = float(np.abs(model.score(fit.params)).max())
    assert gradient < 1e-6, gradient
    return np.asarray(fit.params, dtype=float), gradient


def comparability(t, s):
    return ((s[:, None] == 1) & (t[:, None] < t[None, :])).astype(float)


def interval(c, k):
    T, var = closed_form(c, k)
    T2, var2 = projection(c, k)
    assert abs(T - T2) < 1e-12 and abs(var - var2) <= 1e-12 * max(var, 1e-300)
    se = float(np.sqrt(var))
    return {"estimate": float(T), "se": se, "lower": float(T - Z975 * se), "upper": float(T + Z975 * se)}


def load(study: Path):
    sur = study / "07_reference_study/results/survival"
    saved = json.loads((sur / "survival_illustration.json").read_text())
    rows = {int(r["subject_id"]): r for r in csv.DictReader(open(sur / "complete_case_inputs.csv"))}
    held = list(csv.DictReader(open(sur / "heldout_scores.csv")))
    assert [int(r["subject_id"]) for r in held] == saved["test_subject_ids"]

    def block(ids):
        X = np.array([[float(rows[i][c]) for c in FEATURES] for i in ids])
        t = np.array([float(rows[i]["time"]) for i in ids])
        s = np.array([int(rows[i]["status"]) for i in ids])
        return X, t, s

    Xtr, ttr, str_ = block(saved["train_subject_ids"])
    Xte, tte, ste = block(saved["test_subject_ids"])
    mean, scale = Xtr.mean(0), Xtr.std(0)
    np.testing.assert_allclose(mean, saved["training_means"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale, saved["training_scales"], rtol=0, atol=1e-12)
    scores = {k: np.array([float(r[k]) for r in held]) for k in
              ("age_sex_component", "ecog_component", "full_score")}
    return saved, (Xtr, ttr, str_), (Xte, tte, ste), mean, scale, scores


def contrasts(c, F, Dsame, Drefit):
    """Net change, movement and pair counts on the comparable pairs c (an indicator matrix)."""
    m = c > 0
    hf, hs, hr = H(F), H(Dsame), H(Drefit)
    changed = m & (hf != hr)
    return {
        "concordance_refit": float(hr[m].mean()),
        "net_change_refit": float(hf[m].mean() - hr[m].mean()),
        "movement_refit": float(np.abs(hf - hr)[m].mean()),
        "changed_pairs_refit": int(changed.sum()),
        "strict_reversals_refit": int((m & (F * Drefit < 0)).sum()),
        "tie_transitions_refit": int((changed & ((F == 0) | (Drefit == 0))).sum()),
        "same_fit_versus_refit_changed_pairs": int((m & (hs != hr)).sum()),
        "same_fit_versus_refit_movement": float(np.abs(hs - hr)[m].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "generated")
    a = ap.parse_args()
    saved, (Xtr, ttr, str_), (Xte, tte, ste), mean, scale, scores = load(a.study)
    Ztr, Zte = (Xtr - mean) / scale, (Xte - mean) / scale

    beta, grad = cox(ttr, str_, Ztr)
    np.testing.assert_allclose(beta, saved["coefficients"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(Zte @ beta, scores["full_score"], rtol=0, atol=1e-8)
    full = scores["full_score"]
    c = comparability(tte, ste)
    m = c > 0
    assert int(c.sum()) == 1653 and len(full) == 69

    F = full[:, None] - full[None, :]
    out = {"method": "refit reduced Cox models on the same training patients; held-out comparable pairs",
           "training_patients": int(len(ttr)), "held_out_patients": int(len(tte)),
           "n_comparable_pairs": int(c.sum()), "full_coefficients": beta.tolist(),
           "full_score_equation_residual": grad, "concordance_full": float(H(F)[m].mean()),
           "interval": "estimate +/- z * se with the Proposition 4 variance; coefficients fixed",
           "bootstrap": {"replicates": REPLICATES, "seed": SEED, "resampling_unit": "training patient",
                         "interval": "percentile 2.5/97.5",
                         "evaluation": "fixed 69 held-out patients and 1653 comparable pairs"}}
    reduced = {}
    for key, (cols, ret, dele) in DELETIONS.items():
        cols = list(cols)
        gamma, g = cox(ttr, str_, Ztr[:, cols])
        refit = Zte[:, cols] @ gamma
        same = scores[ret]
        np.testing.assert_allclose(same, Zte[:, cols] @ beta[cols], rtol=0, atol=1e-8)
        Dsame, Drefit = same[:, None] - same[None, :], refit[:, None] - refit[None, :]
        E = scores[dele][:, None] - scores[dele][None, :]
        r = contrasts(c, F, Dsame, Drefit)
        r.update({
            "kept_features": [FEATURES[k] for k in cols],
            "refit_coefficients": gamma.tolist(),
            "same_fit_coefficients": beta[cols].tolist(),
            "refit_score_equation_residual": g,
            "concordance_same_fit": float(H(Dsame)[m].mean()),
            "net_change_same_fit": float(H(F)[m].mean() - H(Dsame)[m].mean()),
            "movement_same_fit": float(np.abs(H(F) - H(Dsame))[m].mean()),
            "budget_same_fit": float(rho(Dsame, E)[m].mean()),
        })
        r["refit_minus_same_fit_net_change"] = r["net_change_refit"] - r["net_change_same_fit"]
        # Proposition 4 intervals with the coefficients held fixed
        k_net = H(F) - H(Drefit)
        r["closed_form"] = {
            "net_change_refit": interval(c, k_net),
            "movement_refit": interval(c, np.abs(k_net)),
            "refit_minus_same_fit_net_change": interval(c, H(Dsame) - H(Drefit)),
        }
        assert abs(r["closed_form"]["net_change_refit"]["estimate"] - r["net_change_refit"]) < 1e-12
        assert abs(r["net_change_refit"]) <= r["movement_refit"] + 1e-12
        assert r["changed_pairs_refit"] == r["strict_reversals_refit"] + r["tie_transitions_refit"]
        out[key] = r
        reduced[key] = cols

    # A one-covariate retained block orders patients identically after refitting whenever
    # its coefficient keeps its sign, so the age-and-sex deletion is unchanged by refitting.
    ag = out["delete_age_sex"]
    if np.sign(ag["refit_coefficients"][0]) == np.sign(ag["same_fit_coefficients"][0]):
        assert ag["same_fit_versus_refit_changed_pairs"] == 0
        assert abs(ag["net_change_refit"] - ag["net_change_same_fit"]) < 1e-12
    # With age and sex retained, both scores order patients of the same sex by age.  A
    # male-female pair is ordered differently only when its age difference, opposing the sex
    # effect, lies between the two age-for-sex trade-offs (in years) of the scores.
    ec = out["delete_ecog"]
    for c in (ec["same_fit_coefficients"], ec["refit_coefficients"]):
        assert c[0] > 0 > c[1], c
    trade = lambda c: abs(c[1] / scale[1]) / abs(c[0] / scale[0])
    ec["age_years_offsetting_sex_same_fit"] = float(trade(ec["same_fit_coefficients"]))
    ec["age_years_offsetting_sex_refit"] = float(trade(ec["refit_coefficients"]))
    lo, hi = sorted((ec["age_years_offsetting_sex_same_fit"], ec["age_years_offsetting_sex_refit"]))
    age, sex = Xte[:, 0], Xte[:, 1]
    dage, dsex = age[:, None] - age[None, :], sex[:, None] - sex[None, :]
    # the sex effect is negative: a male (sex 1) scores higher than a female (sex 2) of equal age
    opposing = (dsex != 0) & (np.sign(dage) == np.sign(dsex)) & m
    between = opposing & (np.abs(dage) > lo) & (np.abs(dage) < hi)
    ec["comparable_pairs_between_tradeoffs"] = int(between.sum())
    assert ec["comparable_pairs_between_tradeoffs"] == ec["same_fit_versus_refit_changed_pairs"]
    ec["largest_opposing_age_difference_comparable"] = float(np.abs(dage[opposing]).max())

    rng = np.random.default_rng(SEED)
    names = ("net_change_same_fit", "movement_same_fit", "budget_same_fit",
             "net_change_refit", "movement_refit")
    draws = {k: {q: [] for q in names} for k in DELETIONS}
    sign_kept = {k: 0 for k in DELETIONS}
    n = len(ttr)
    for _ in range(REPLICATES):
        idx = rng.integers(0, n, n)
        b, _g = cox(ttr[idx], str_[idx], Ztr[idx])
        sb = Zte @ b
        Fb = sb[:, None] - sb[None, :]
        for key, (cols, _ret, _dele) in DELETIONS.items():
            cols = list(cols)
            gb, _g = cox(ttr[idx], str_[idx], Ztr[idx][:, cols])
            lb = Zte[:, cols] @ b[cols]
            rb = Zte[:, cols] @ gb
            Db, Rb = lb[:, None] - lb[None, :], rb[:, None] - rb[None, :]
            Eb = Fb - Db
            hf, hs, hr = H(Fb)[m], H(Db)[m], H(Rb)[m]
            draws[key]["net_change_same_fit"].append(float(hf.mean() - hs.mean()))
            draws[key]["movement_same_fit"].append(float(np.abs(hf - hs).mean()))
            draws[key]["budget_same_fit"].append(float(rho(Db, Eb)[m].mean()))
            draws[key]["net_change_refit"].append(float(hf.mean() - hr.mean()))
            draws[key]["movement_refit"].append(float(np.abs(hf - hr).mean()))
            sign_kept[key] += int(np.all(np.sign(gb) == np.sign(b[cols])))
    for key in DELETIONS:
        boot = {}
        for q in names:
            v = np.array(draws[key][q])
            lo, hi = np.percentile(v, [2.5, 97.5])
            boot[q] = {"lower": float(lo), "upper": float(hi), "sd": float(v.std(ddof=1)), "mean": float(v.mean())}
        v = np.array(draws[key]["movement_same_fit"]) - np.abs(np.array(draws[key]["net_change_same_fit"]))
        boot["movement_minus_absolute_net_same_fit"] = {"lower": float(np.percentile(v, 2.5)),
                                                        "upper": float(np.percentile(v, 97.5)),
                                                        "fraction_positive": float(np.mean(v > 0))}
        boot["reduced_coefficient_signs_kept"] = sign_kept[key] / REPLICATES
        out[key]["training_bootstrap"] = boot

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "survival_refit.json").write_text(json.dumps(out, indent=1))
    print(f"full C {out['concordance_full']:.4f}; {out['n_comparable_pairs']} comparable pairs")
    for key in DELETIONS:
        r = out[key]
        print(f"{key}: same-fit C {r['concordance_same_fit']:.4f} net {r['net_change_same_fit']:+.4f} "
              f"A {r['movement_same_fit']:.4f} | refit C {r['concordance_refit']:.4f} "
              f"net {r['net_change_refit']:+.4f} A {r['movement_refit']:.4f} "
              f"rev {r['strict_reversals_refit']} ties {r['tie_transitions_refit']} | "
              f"same vs refit changed {r['same_fit_versus_refit_changed_pairs']}")
        tb = r["training_bootstrap"]
        print("   training bootstrap: " + "; ".join(
            f"{q} [{tb[q]['lower']:.3f}, {tb[q]['upper']:.3f}]" for q in names)
            + f"; signs kept {tb['reduced_coefficient_signs_kept']:.3f}"
            + f"; A-|Delta|>0 in {tb['movement_minus_absolute_net_same_fit']['fraction_positive']:.3f}")


if __name__ == "__main__":
    main()
