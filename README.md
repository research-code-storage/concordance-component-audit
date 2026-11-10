# Reproducing the component-concordance analysis

This repository reproduces the numerical results of *Concordance changes in additive predictors: decomposition, bounds and inference*. The result locations refer to the accompanying manuscript and Supporting Information.

The starting inputs are the supplied feature matrices, patient data, fitted model parameters, component scores and simulation records. Results are written to `generated/`. The supplied inputs remain unchanged.

## Run the analysis

Use Python 3.10. The numerical dependencies are pinned in `requirements.txt`.

```sh
python -m venv ../concordance-env
# Activate the environment, then:
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows PowerShell, activate with `../concordance-env/Scripts/Activate.ps1`. On Linux/macOS, use `source ../concordance-env/bin/activate`.

The command checks file integrity, reproduces the analyses, writes the numerical tables and figures, and compares the results with `reported/`. Successful completion produces `generated/verification.json` with `"status": "PASS"`.

To additionally recompute the component scores from the supplied feature matrices and all 94 retained fitted models, reconstruct the frozen coordinate assets, and repeat the small-problem gradient checks:

```sh
python -m pip install -r requirements-training.txt
python reproduce.py --deep
```

## Find a specific result

| Result | Location in the paper | Where to look |
|---|---|---|
| Evaluation panel sizes | Table 4 | `generated/tables/Table_Main4.csv` |
| Simulation means and ratios | Section 3, Table 2, Tables S1-S3 | `generated/simulation_usefulness.json`, `generated/fitted_component_audit.json` |
| Clinical component analysis | Section 4.1, Table 3 | `generated/survival_example.json`, `generated/lung_bootstrap.json`, `generated/lung_ustat.json` |
| Clinical refitting and training variability | Section 4.1, Tables S12-S13 | `generated/survival_refit.json` |
| Toxicogenomic AUCs and bounds | Section 4.2, Figure 2, Tables S4-S5 | `generated/results.json`, `generated/main_inference/summary.json` |
| Permutation comparisons | Section 4.2, Table S6 | `generated/permutation_contrast.json` |
| Frozen-reference sensitivity | SI H.3 | `generated/frozen_reference.json`, `generated/frozen_component_audit.json` |
| Complete-rank comparison | SI H.4, Table S7 | `generated/theory_replay/complete_rank_diagnostics.csv`, `generated/tables/Table_S7.csv` |
| Bounded nonlinear component perturbations | Section 4.2, SI H.5, Table S8 | `generated/nonlinear/` |
| Interval coverage and standard-error calibration | Section 3, SI I, Tables S9-S11 | `data/evidence/08_inference/`, `generated/inference_verification.json` |
| Figures 1, 2 and S1 | Main text and SI C | `generated/figures/Figure_1.pdf`, `Figure_2.pdf`, `Figure_S1.pdf` |

`generated/key_results.csv` gives 87 key values with their manuscript locations, result files and exact JSON field paths. `reported/table_numeric_rows.json` contains the 285 published numerical rows against which the regenerated tables are checked. Main Table 1 and SI Table S14 summarize definitions and comparisons and have no experimental numbers to regenerate.

## Detailed instructions

- [REPRODUCTION.md](REPRODUCTION.md): commands, calculations, expected results and full regeneration options.
- [docs/PARAMETERS.md](docs/PARAMETERS.md): model settings, designs, seeds and numerical conventions.
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md): file layout, units, identifiers and array dimensions.
- [docs/NUMERICAL_MAP.md](docs/NUMERICAL_MAP.md): direct links from key reported values to computed fields.

For integrity checking alone, run `python reproduce.py --manifest-only`. For one analysis, use `python reproduce.py --stage clinical`, replacing `clinical` with a stage listed in the walkthrough. The `report` stage combines outputs from the preceding stages.

The original input files are under `data/evidence/`. The nonlinear-test inputs and parameters are under `b_robustness/`. The software license is in `LICENSE`; third-party data and software terms are described in `LICENSE.md`.
