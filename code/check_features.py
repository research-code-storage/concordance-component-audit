#!/usr/bin/env python3
"""Check feature dimensions, sample identities, duplicate vectors, label consistency and matrix alignment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_matrix(path: Path):
    """Return (sample_ids, float matrix) from a wide CSV whose first column is the id."""
    with open(path) as fh:
        r = csv.reader(fh)
        header = next(r)
        ids, rows = [], []
        for line in r:
            ids.append(line[0])
            rows.append([float(x) for x in line[1:]])
    return ids, np.asarray(rows), header[1:]


def duplicate_groups(mat: np.ndarray):
    """Indices of rows that are bit-identical to an earlier row."""
    seen, dups = {}, []
    for i, row in enumerate(mat):
        key = row.tobytes()
        if key in seen:
            dups.append((seen[key], i))
        else:
            seen[key] = i
    return dups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "generated")
    a = ap.parse_args()

    d = a.release / "05_reconstruction/data"
    report, failures = {}, []

    matrices = {}
    for name, fname in (("mhedili pathways", "mhedili_pathways.csv"),
                        ("mhedili embeddings", "mhedili_embeddings.csv"),
                        ("DILImap pathways", "dilimap_pathways.csv"),
                        ("DILImap embeddings", "dilimap_embeddings.csv")):
        ids, mat, cols = read_matrix(d / fname)
        matrices[name] = (ids, mat)
        dups = duplicate_groups(mat)
        unique_ids = len(set(ids)) == len(ids)
        report[name] = {"rows": len(ids), "columns": mat.shape[1],
                        "duplicate_rows": len(dups), "identifiers_unique": unique_ids}
        print(f"{name:20s} {len(ids):5d} rows x {mat.shape[1]:4d}  "
              f"duplicate rows: {len(dups)}  ids unique: {unique_ids}")
        if dups:
            failures.append(f"{name}: {len(dups)} duplicate feature vectors")
        if not unique_ids:
            failures.append(f"{name}: sample identifiers are not unique")

    # expected sizes, as stated in Supplementary Information E
    for name, expected in (("mhedili pathways", 7818), ("mhedili embeddings", 7818),
                           ("DILImap pathways", 763), ("DILImap embeddings", 763)):
        got = report[name]["rows"]
        if got != expected:
            failures.append(f"{name}: expected {expected} rows, found {got}")

    # pathway and embedding tables must describe the same samples
    for coll in ("mhedili", "DILImap"):
        p = set(matrices[f"{coll} pathways"][0])
        e = set(matrices[f"{coll} embeddings"][0])
        if p != e:
            failures.append(f"{coll}: pathway and embedding sample sets differ "
                            f"({len(p - e)} / {len(e - p)} unmatched)")
        else:
            print(f"{coll:20s} pathway and embedding tables cover the same samples")

    # no feature vector shared across the two collections
    for kind in ("pathways", "embeddings"):
        m = matrices[f"mhedili {kind}"][1]
        x = matrices[f"DILImap {kind}"][1]
        if m.shape[1] == x.shape[1]:
            seen = {row.tobytes() for row in m}
            shared = sum(1 for row in x if row.tobytes() in seen)
            report[f"cross-collection {kind}"] = {"shared_vectors": shared}
            print(f"{'cross-collection ' + kind:20s} shared vectors: {shared}")
            if shared:
                failures.append(f"{kind}: {shared} vectors shared across collections")

    # labels must be constant within a compound
    meta = list(csv.DictReader(open(d / "dilimap_metadata.csv")))
    by_compound = {}
    for row in meta:
        by_compound.setdefault(row["chemical"], set()).add(row["labels"])
    conflicts = {k: v for k, v in by_compound.items() if len(v) > 1}
    report["DILImap label conflicts"] = len(conflicts)
    print(f"{'DILImap compounds':20s} {len(by_compound)} with "
          f"{len(conflicts)} within-compound label conflicts")
    if conflicts:
        failures.append(f"within-compound label conflicts: {sorted(conflicts)[:5]}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "feature_checks.json").write_text(json.dumps(report, indent=1))

    if failures:
        print("\nFAILURES")
        for f in failures:
            print(" -", f)
        return 1
    print("\nFEATURE-LEVEL CLAIMS CONFIRMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
