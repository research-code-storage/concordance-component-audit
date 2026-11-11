#!/usr/bin/env python3
"""Recompute the lung-cancer survival illustration from the per-patient scores.

Independent of the study package's own summary: comparable pairs are re-enumerated from
`time` and `status`, and concordance and the Proposition 1 bounds are computed with the
same functions used for the toxicogenomic panels (compute_all.py).  The saved
`survival_illustration.json` is read only at the end, to compare.

Comparable pairs are ordered pairs (i, j) with status_i == 1 and time_i < time_j, each
counted once; equal times are excluded.  All weights are 1 (no censoring weighting).
A higher score means higher risk of an earlier event, so the pair contributes
H(score_i - score_j) with H(d) = 1{d>0} + 0.5*1{d=0}.

Usage:  python code/survival_example.py --study data/evidence
Writes: generated/survival_example.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# NOTE: compute_all.pair_budget takes positive/negative score vectors and forms their
# cross product, which is right for a binary panel but wrong here: survival comparable
# pairs are enumerated explicitly, so the bound must be evaluated pair by pair.


def pair_rho(D, E):
    """Per-pair maximum absolute contribution change, elementwise over enumerated pairs."""
    import numpy as _np
    aD, aE = _np.abs(D), _np.abs(E)
    rho = _np.zeros_like(D, dtype=float)
    nz = D != 0
    rho[nz & (aE > aD)] = 1.0
    rho[nz & (aE == aD)] = 0.5
    rho[(~nz) & (E != 0)] = 0.5
    return rho


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "id": [r["subject_id"] for r in rows],
        "time": f("time"),
        "status": f("status").astype(int),
        "base": f("age_sex_component"),
        "added": f("ecog_component"),
        "full": f("full_score"),
    }


def comparable(time, status):
    """Ordered index pairs (i, j): i has an event strictly before j is last seen."""
    n = len(time)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ok = (status[I] == 1) & (time[I] < time[J])
    return I[ok], J[ok]


def concordance(diff):
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "generated")
    args = ap.parse_args()

    sur = args.study / "07_reference_study/results/survival"
    d = load(sur / "heldout_scores.csv")
    i, j = comparable(d["time"], d["status"])
    F = d["full"][i] - d["full"][j]
    c_full = concordance(F)
    h_full = (F > 0) + 0.5 * (F == 0)
    npairs = int(len(F))

    # Both deletions come from the same fitted Cox model.  Deleting the other component
    # is not a second experiment: it is the same decomposition read the other way, and
    # the two directions behave very differently.
    def deletion(retained: str, deleted: str) -> dict:
        D = d[retained][i] - d[retained][j]
        E = d[deleted][i] - d[deleted][j]
        c_ret = concordance(D)
        h_ret = (D > 0) + 0.5 * (D == 0)
        abs_pair = float(np.abs(h_full - h_ret).mean())
        changed = int((h_full != h_ret).sum())
        reversals=int(((D*F)<0).sum())
        tie_transitions=int(((h_full!=h_ret)&((D==0)|(F==0))).sum())
        positive_budget_pairs=int((pair_rho(D,E)>0).sum())
        # Proposition 2: untied pairs with abs(E_ij) > abs(D_ij)
        exceeds=(D!=0)&(np.abs(E)>np.abs(D))
        reinforced=int((exceeds&(D*E>0)).sum());opposed=int((exceeds&(D*E<0)).sum())
        assert opposed==reversals
        assert changed==reversals+tie_transitions
        budget = float(pair_rho(D, E).mean())
        vulnerable = int((np.abs(D) <= np.abs(E)).sum())
        net = c_full - c_ret
        return {
            "concordance_retained": c_ret,
            "net_change": net,
            "mean_absolute_pair_change": abs_pair,
            "tie_aware_bound": budget,
            "inclusive_bound": vulnerable / npairs,
            "changed_pairs": changed,
            "strict_reversals": reversals,
            "tie_transitions": tie_transitions,
            "positive_budget_pairs": positive_budget_pairs,
            "reinforced_untied_pairs": reinforced,
            "opposed_untied_pairs": opposed,
            "vulnerable_pairs": vulnerable,
            "hidden_movement_ratio": abs_pair / abs(net),
            "slack_bound_over_movement": budget / abs_pair,
        }

    out = {
        "n_patients": len(d["id"]),
        "n_events": int(d["status"].sum()),
        "n_comparable_pairs": npairs,
        "concordance_full": c_full,
        "delete_ecog": deletion("base", "added"),
        "delete_age_sex": deletion("added", "base"),
    }
    # the manuscript's primary figures refer to the ECOG deletion
    out.update({k: v for k, v in out["delete_ecog"].items()})

    print(f"patients {out['n_patients']}  events {out['n_events']}  "
          f"comparable pairs {npairs}   C(full) = {c_full:.6f}")
    for key, label in (("delete_ecog", "delete ECOG"),
                       ("delete_age_sex", "delete age and sex")):
        r = out[key]
        print(f"\n  {label}")
        print(f"    C(retained)        {r['concordance_retained']:.6f}")
        print(f"    net change         {r['net_change']:+.6f}")
        print(f"    mean |pair change| {r['mean_absolute_pair_change']:.6f}  "
              f"({r['hidden_movement_ratio']:.2f}x the net)")
        print(f"    tie-aware bound    {r['tie_aware_bound']:.6f}  "
              f"({r['slack_bound_over_movement']:.2f}x the movement)")
        print(f"    changed pairs      {r['changed_pairs']} of {npairs}")
        print(f"    flagged pairs      {r['vulnerable_pairs']} of {npairs}")

    # compare against the study package's own saved summary
    arch = json.loads((sur / "survival_illustration.json").read_text())
    e = out["delete_ecog"]
    mapping={'net_change':arch['pair_bound']['signed_change'],
             'mean_absolute_pair_change':arch['pair_bound']['absolute_pair_change'],
             'tie_aware_bound':arch['pair_bound']['tie_aware_bound'],
             'concordance_retained':arch['base_C']}
    for key,value in mapping.items():
        np.testing.assert_allclose(e[key],value,rtol=0,atol=1e-12)
    np.testing.assert_allclose(c_full,arch['full_C'],rtol=0,atol=1e-12)
    assert out['delete_age_sex']['strict_reversals']==240
    assert out['delete_age_sex']['tie_transitions']==572
    assert out['delete_age_sex']['positive_budget_pairs']==974
    print('Saved primary deletion and complementary tie/reversal counts verified.')

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "survival_example.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
