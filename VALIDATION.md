# Validation record

The v41 package was checked on 8 November 2026 with Python 3.10.19 and the pinned numerical dependencies. `python reproduce.py` completed from an empty `generated/` directory.

- All 11 workflow stages passed.
- All 16 numerical tables, 285 table rows and 87 key numerical targets matched.
- The 14,400 movement/slack datasets, 1,000 clinical training bootstrap fits and 60,000 permutation drops were regenerated and checked.
- All 160,000 stored inference estimates were reaggregated. Three datasets per scenario and one bootstrap dataset per selected scenario were regenerated, giving 240 dataset checks and eight bootstrap checks. The instructions also give the command for full regeneration.
- Scores and objectives for 94 saved models were independently recalculated from the supplied feature matrices and fitted parameters. Maximum score/parameter and objective differences were 1.89e-15 and 7.11e-15.
- Fifteen synthetic small-problem fits and three real-data fits agreed with independent gradient-based implementations. The frozen reference assets were also reconstructed from the feature matrices.
- All 24,240 nonlinear grid evaluations and their maximizing witnesses were checked.
- Figures 1, 2 and S1 were generated and visually inspected.

`VALIDATION.json` gives the machine-readable counts, errors and environment. `MANIFEST.sha256` identifies every distributed file. The default workflow uses the supplied feature matrices, model states and simulation records; fresh full fitting and full inference-simulation commands are documented in `REPRODUCTION.md`.
