#!/usr/bin/env python3
"""Compute component-score permutation and deletion comparisons.

For each fitted model, hold the retained score fixed, permute the component score
across compounds, and calculate the mean concordance drop. Calculate the deletion
bound and changed-pair counts from the same fitted decomposition.

Usage: python code/permutation_contrast.py --release data/evidence
Outputs: generated/permutation_contrast.json and generated/table7.tex.
"""
from __future__ import annotations

import argparse
import json,hashlib
from pathlib import Path

import numpy as np

from compute_all import load_panel, concordance, pair_budget, actual_change

DESIGNS = [
    ("historical_centred", "Pooled, centred"),
    ("matched_centred", "Matched, centred"),
    ("revised_origin", "Matched, origin preserving"),
]
PANELS = (("PHH", "PHH"), ("external", "DILImap"))
ARM = "softplus_original"
SEEDS = range(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "generated")
    ap.add_argument('--write-reference', action='store_true', help='Write a reference record for the configured permutation run')
    a = ap.parse_args()

    exp = a.release / "05_reconstruction/results/experiments"
    out = {}
    draw_drops = {}
    records = []
    source_hashes = {}
    seed_offsets = json.loads((exp / "config.json").read_text())["permutation_seed_offsets"]

    for domain, label in PANELS:
        per_seed = [load_panel(exp, 60, s, domain) for s in SEEDS]
        y = per_seed[0]["y"]
        pos, neg = y == 1, y == 0

        for design, dname in DESIGNS:
            arm = f"{design}__{ARM}"
            k = per_seed[0]["arms"].index(arm)

            cert_bound, cert_changed, perm_imp, perm_nonzero, deletion = [], [], [], [], []
            for si, p in enumerate(per_seed):
                full = p["full"][:, k]
                lin = p["linear"][:, k]
                ker = p["kernel"][:, k]

                b, v, npairs = pair_budget(lin[pos], lin[neg], ker[pos], ker[neg])
                d, changed = actual_change(lin[pos], lin[neg], ker[pos], ker[neg])
                cert_bound.append(b)
                cert_changed.append(changed)
                deletion.append(d)

                base = concordance(full[pos], full[neg])
                rng = np.random.default_rng(a.seed + 100 * si + seed_offsets[arm])
                drops = np.empty(a.permutations)
                for t in range(a.permutations):
                    permuted = lin + ker[rng.permutation(len(ker))]
                    drops[t] = base - concordance(permuted[pos], permuted[neg])
                name=f'{domain}__{design}__seed{si}'
                draw_drops[name]=drops.copy()
                scorefile=exp/f'n60_seed{si}_{domain}_scores.npz'
                source_hashes[scorefile.name]=hashlib.sha256(scorefile.read_bytes()).hexdigest()
                records.append({'name':name,'rng_seed':a.seed+100*si+seed_offsets[arm],'arm_index':k,'compound_order':p['compounds']})
                perm_imp.append(float(drops.mean()))
                perm_nonzero.append(float((drops != 0).mean()))

            cell = {
                "panel": label, "design": dname, "pairs": npairs,
                "certificate_bound_max": float(max(cert_bound)),
                "certificate_changed_max": int(max(cert_changed)),
                "deletion_difference_mean": float(np.mean(deletion)),
                "permutation_importance_mean": float(np.mean(perm_imp)),
                "permutation_importance_max": float(np.max(perm_imp)),
                "permutation_nonzero_fraction": float(np.mean(perm_nonzero)),
            }
            out[f"{domain}|{design}"] = cell
            print(f"{label:8s} {dname:26s} bound<={cell['certificate_bound_max']:.4f}  "
                  f"changed<={cell['certificate_changed_max']:<3d}  "
                  f"deletion {cell['deletion_difference_mean']:+.4f}  "
                  f"perm.imp {cell['permutation_importance_mean']:+.4f}  "
                  f"(nonzero on {100*cell['permutation_nonzero_fraction']:.0f}% of permutations)")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "permutation_contrast.json").write_text(json.dumps(out, indent=1))

    np.savez_compressed(a.out/'permutation_drops.npz', **draw_drops)
    config={'seed':a.seed,'permutations_per_draw':a.permutations,'training_compounds':60,'source_hashes':source_hashes,'draws':records}
    (a.out/'permutation_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    reference=a.release/'08_permutation'
    if a.write_reference:
        reference.mkdir(exist_ok=True)
        import shutil
        for name in ['permutation_contrast.json','permutation_drops.npz','permutation_config.json']:
            shutil.copy2(a.out/name, reference/name)
    else:
        from replay_checks import compare
        compare(out,json.loads((reference/'permutation_contrast.json').read_text()),tol=1e-12)
        assert config==json.loads((reference/'permutation_config.json').read_text())
        saved=np.load(reference/'permutation_drops.npz')
        assert set(saved.files)==set(draw_drops)
        for name,values in draw_drops.items():np.testing.assert_allclose(values,saved[name],rtol=0,atol=1e-12)
        print('All 60000 permutation drops and the configuration agree with the released reference.')

    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Component-score permutation and the deletion bound on the same "
         r"fitted states, for the full predictor with 60 training compounds. "
         r"\emph{Bound} and \emph{changed pairs} are the tie-aware bound of "
         r"Proposition~\ref{prop:bound} and the pairs whose contribution actually changes "
         r"on deletion, both maxima over the ten source draws. \emph{Deletion} is the mean "
         r"signed concordance change. \emph{Permutation} is the mean drop in concordance "
         rf"when the kernel component is permuted across compounds, over {a.permutations} permutations "
         rf"per draw, and the last column is the fraction of permutations that change "
         r"concordance at all. Permutation evaluates recombined component scores; "
         r"the bound controls deletion at the observed component alignment.}",
         r"\label{tab:permutation}",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"Panel & Coordinate design & Bound & Changed & Deletion & Permutation & "
         r"Nonzero \\", r"\midrule"]
    for key, c in out.items():
        L.append(f"{c['panel']} & {c['design']} & {c['certificate_bound_max']:.4f} & "
                 f"{c['certificate_changed_max']} & {c['deletion_difference_mean']:+.4f} & "
                 f"{c['permutation_importance_mean']:+.4f} & "
                 f"{100*c['permutation_nonzero_fraction']:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (a.out / "table7.tex").write_text("\n".join(L) + "\n")
    print(f"\nwrote {a.out/'table7.tex'}")


if __name__ == "__main__":
    main()
