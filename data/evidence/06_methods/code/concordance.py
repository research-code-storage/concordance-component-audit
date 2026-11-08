"""Finite-sample component deletion bounds for fixed weighted comparable pairs.

Weights and pairs must be held fixed when scores change. No fitting, outcome
resampling, distribution assumption, or population confidence claim is involved.
"""
import numpy as np


def contribution(d):
    d = np.asarray(d, dtype=float)
    return (d > 0).astype(float) + .5 * (d == 0)


def pair_bound(base_difference, added_difference, weights=None):
    d, e = np.broadcast_arrays(np.asarray(base_difference, float),
                               np.asarray(added_difference, float))
    w = np.ones_like(d) if weights is None else np.broadcast_to(np.asarray(weights, float), d.shape)
    if not all(np.isfinite(x).all() for x in (d, e, w)) or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Finite differences and nonnegative weights of positive total are required')
    ad, ae = np.abs(d), np.abs(e)
    rho = ((ad > 0) & (ad < ae)).astype(float)
    rho += .5 * (((ad == ae) & (ad > 0)) | ((ad == 0) & (ae > 0)))
    change = contribution(d + e) - contribution(d)
    return dict(signed_change=float(np.sum(w * change) / w.sum()),
                absolute_pair_change=float(np.sum(w * np.abs(change)) / w.sum()),
                tie_aware_bound=float(np.sum(w * rho) / w.sum()),
                inclusive_bound=float(np.sum(w * (ad <= ae)) / w.sum()),
                changed_pairs=int(np.count_nonzero(change)),
                vulnerable_pairs=int(np.count_nonzero(ad <= ae)),
                pair_count=int(d.size), weight_sum=float(w.sum()))


def comparable_pairs(times, events):
    """Harrell-type pairs: earlier observed event versus a later follow-up.

    Equal follow-up times are excluded. Larger scores predict earlier events.
    """
    t, event = np.asarray(times, float), np.asarray(events)
    if t.ndim != 1 or event.shape != t.shape or not np.isfinite(t).all() or not np.isin(event, [0, 1]).all():
        raise ValueError('Expected finite follow-up times and binary event indicators')
    return np.nonzero(event[:, None].astype(bool) & (t[:, None] < t[None, :]))
