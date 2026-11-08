"""Verify the SI G.3 explanation against every supplied source-coordinate asset.

No model is refitted. Outputs go to generated/, leaving released records intact.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / 'data/evidence'
sys.path.insert(0, str(EV / '05_reconstruction/code'))
import reconstructed_features as implementation


def main():
    rule = json.loads((ROOT / 'prototype_selection.json').read_text(encoding='utf-8'))
    assert rule['keywords'] == implementation.KEYWORDS
    mi = [i for i, n in enumerate(implementation.PC)
          if any(k in n for k in implementation.KEYWORDS)]
    assert mi == rule['pathway_indices_zero_based']
    assert len(mi) == 16
    assert [implementation.PC[i] for i in mi] == rule['pathway_names']
    checks = []
    rng = np.random.default_rng(20261021)
    u = rng.normal(size=(31, 54))
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    def check(name, asset):
        c = np.asarray(asset['centroids'], dtype=float)
        assert c.shape == (4, 54)
        np.testing.assert_allclose(np.linalg.norm(c, axis=1), 1, atol=1e-12, rtol=0)
        s = c[:, mi].mean(axis=1)
        assert s[0] >= s[1] and s[2] <= s[3], (name, s)
        actual = u @ (c[1] - c[3])
        separate_cosines = u @ c[1] - u @ c[3]
        np.testing.assert_allclose(actual, separate_cosines, atol=1e-14, rtol=0)
        checks.append(dict(asset=name, metabolic_summaries=s.tolist(),
                           selected_positions_zero_based=[1, 3],
                           maximum_cosine_identity_error=float(np.max(np.abs(actual-separate_cosines)))))

    for p in sorted((EV / '05_reconstruction/results/experiments').glob('n*_seed*.json')):
        record = json.loads(p.read_text(encoding='utf-8'))
        for name, asset in record['coordinate_assets'].items():
            check(p.stem + '/' + name, asset)
    assert len(checks) == 30, len(checks)
    frozen = json.loads((EV / '07_reference_study/assets/frozen_reference.json').read_text())
    for name, asset in frozen['recipes'].items():
        assert asset['metabolic_indices'] == mi
        check('frozen/' + name, asset)
    assert len(checks) == 31, len(checks)
    report = dict(status='PASS', draw_specific_assets=30, frozen_assets=1,
                  metabolic_pathway_count=16, checks=checks)
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    (out / 'prototype_selection_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('PASS: all 30 draw-specific assets and the frozen-reference asset match the stated ordering;')
    print('16 pathway names and the cosine-difference identity verified.')


if __name__ == '__main__':
    main()
