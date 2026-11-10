# Reproducing the reported numbers

All commands below run from the repository root. `python reproduce.py` executes the stages in the order shown. A failed numerical or integrity check exits with a nonzero status. Each subprocess writes its complete output to `generated/logs/`.

## 1. Inputs and file integrity

`MANIFEST.sha256` covers the distributed code, configurations, inputs and numerical targets. The entry point verifies every file before calculation. `generated/` contains new results and is excluded from the manifest.

```sh
python reproduce.py --manifest-only
python reproduce.py --stage inputs
```

The input stage checks four feature matrices, metadata alignment, unique sample identifiers, finite values, compound labels and the train/evaluation separation. The matrices have 7,818 MHEDILI and 763 DILImap profiles, each with 768 embedding features and 54 pathway features. The analysis selects the source and PHH panels described in SI G. Other collection rows remain in the matrices so the collection counts and sample identifiers can be checked directly.

The three evaluation panels contain 1,458 profiles from 20 within-source compounds, 583 profiles from 49 DILImap compounds, and 227 profiles from 11 PHH compounds. They give 96, 528 and 24 cross-class compound pairs, respectively.

## 2. Toxicogenomic scores, AUCs and intervals

```sh
python reproduce.py --stage toxicogenomic
```

Inputs are the ten `n60_seed*_state.npz` files and their panel score files under `data/evidence/05_reconstruction/results/experiments/`. Each draw contains the full predictor and kernel-only model under three reference designs. The unfitted sum of the two coordinates is stored in the diagnostic columns. `config.json` names the models and reference groups.

`code/compute_all.py` averages profile scores within compound, forms all positive-negative compound pairs, and evaluates

```text
H(x) = 1(x > 0) + 0.5 1(x = 0)
D_ij = retained_i - retained_j
E_ij = component_i - component_j
Delta = mean[H(D_ij + E_ij) - H(D_ij)]
A     = mean[abs(H(D_ij + E_ij) - H(D_ij))]
B     = mean[max(abs(H(D_ij + abs(E_ij)) - H(D_ij)),
                 abs(H(D_ij - abs(E_ij)) - H(D_ij)))]
```

The point AUC is calculated separately in each draw and then averaged over ten draws. For intervals, `analyse_reconstructed.py` generates 20,000 compound resamples within outcome class with seed 20260922. The same resample is applied across draws and prediction scores. It averages draw-specific pair contributions and uses the 2.5th and 97.5th percentiles of the resulting AUCs. The intervals therefore describe compound sampling conditional on the supplied fits.

Expected full-predictor results under `matched_centred__softplus_original` are:

| Panel | Mean AUC | Conditional 95% interval | Maximum B at t = 1 |
|---|---:|---|---:|
| PHH | 0.8750 | [0.6250, 1.0000] | 0 |
| DILImap | 0.6064 | [0.4375, 0.7687] | 0.0076 |
| Within-source holdout | 0.4292 | [0.1771, 0.6969] | 0.0833 |

`generated/results.json` contains per-draw values as well as maxima and ranges. `generated/main_inference/summary.json` contains all 27 panel/design/score cells and their intervals. For SI H.3, the matched-minus-pooled AUC difference on DILImap is taken from the paired bootstrap contrast in this summary, not by subtracting endpoints of marginal intervals.

## 3. Component-score permutation

```sh
python reproduce.py --stage permutation
```

`code/permutation_contrast.py` holds the compound-level linear score fixed, permutes the kernel score across compounds, and recomputes concordance. There are 1,000 permutations for each of ten draws, three reference designs and two panels, giving 60,000 drops. The explicit per-draw random seeds are in `data/evidence/08_permutation/permutation_config.json`. The `permutation_seed_offsets` in the fitting configuration preserve these streams independently of the model-column positions.

The matched-centred mean deletion and permutation drops are 0 and 0.0082 on PHH, and 0.0017 and 0.0015 on DILImap. The script compares every regenerated permutation drop with the supplied reference array.

## 4. Clinical analysis and two sources of variation

```sh
python reproduce.py --stage clinical
```

The patient snapshot is `data/evidence/07_reference_study/data/lung.csv`. In this supplied snapshot, `status = 1` denotes death and `status = 0` censoring. The required variables are time, status, age, sex and ECOG. The single record missing ECOG is excluded, leaving 227 patients. The permutation from NumPy's `default_rng(20260928)` assigns the first 158 to training and the remaining 69 to evaluation. The split is unstratified. Covariates are standardized using training means and standard deviations with divisor 158.

The Cox fit uses Efron handling of tied events. A comparable pair has an observed event in patient i strictly before patient j's follow-up time. The 69 evaluation patients produce 1,653 such pairs, each with unit weight. Higher Cox scores indicate earlier events.

The stage refits the full Cox model, checks the saved patient scores, and independently enumerates the pair contributions. Deleting age and sex gives Delta = 0.0090744102, A = 0.3182093164 and B = 0.4162129462, with 240 strict reversals and 572 transitions to ties. Deleting ECOG gives Delta = 0.0695704779, A = 0.1954022989 and B = 0.3986690865, with 316 reversals and 14 tie transitions.

`lung_bootstrap.py` uses 5,000 evaluation-patient resamples with seed 20261005, holding the fitted coefficients fixed and rebuilding comparable pairs. `lung_ustat.py` calculates the closed-form variance of Proposition 4 and uses 1.959963984540054 standard errors for the normal interval. Paired deletion contrasts are formed from differences of their kernels on the same pairs.

`survival_refit.py` separately draws 1,000 training-patient bootstrap samples with seed 20261019. It refits the full and both reduced models in each sample and evaluates all comparisons on the original evaluation patients. The same-fit analysis removes a block from that replicate's full fit. The refit comparison uses the full and newly fitted reduced scores. The output provides the intervals in Table S13, including [0.173, 0.450] for same-fit age-and-sex movement.

## 5. Movement and bound slack simulation

```sh
python reproduce.py --stage simulation
```

This runs `simulate_bounds.py` and compares all 14,400 replicate rows against the saved CSV. The 72 scenarios cross n = 50 or 200, four component strengths, three component geometries and three retained-score distributions. There are 200 replicates per scenario, with replicate seeds generated from master seed 20260927.

The aggregation in `simulation_usefulness.py` distinguishes the mean of replicate-level ratios from the ratio of means. Table 2 uses mean(A / abs(Delta)) among replicates with nonzero Delta and mean(B / A) among replicates with positive A. The valid denominators are recorded in the output. The geometry-specific ratios of mean A to mean absolute Delta are approximately 3.19, 4.48 and 1.82 for aligned, independent and opposed components.

## 6. Frozen-reference labelled-sample sensitivity

```sh
python reproduce.py --stage frozen_reference
```

The fixed source reference is in `07_reference_study/assets/frozen_reference.json`. There are 31 saved full-predictor fits: ten draws each at 4, 20 and 60 labelled compounds, and one fit using all 102 compounds. The reference, its scales and the embedding scale remain fixed across these training sizes. The supervised hinge weights are compound-balanced, C / (2 N_y m_g).

`frozen_reference.py` recomputes mean compound AUCs from the per-profile scores and the component analysis at 60 labelled compounds. DILImap AUCs are 0.6348, 0.6333, 0.6331 and 0.6326. PHH AUCs are 0.7458, 0.7250, 0.7125 and 0.7083. These are the sensitivity results in SI H.3.

## 7. Fitted-model simulation

```sh
python reproduce.py --stage fitted_simulation
```

The 200 archives `04_results/extension_double/simulation_rep*.npz` contain the fitted parameters and scores. For each training draw there are two coordinate types, two initializations and three evaluation displacements. `replay_checks.py fitted_sim` recalculates all 2,400 AUC, Delta, A and B evaluations from those scores, then calculates means and Monte Carlo standard errors over the 200 training draws. The evaluation sample and fitted models are reused across the three displacements.

For noise coordinates and zero initialization at displacement 16, the means are Delta = 0.0578, A = 0.1198 and B = 0.1223. Figure 1 uses two Monte Carlo standard errors for its error bars.

## 8. Theory calculations and complete rankings

```sh
python reproduce.py --stage theory
```

This regenerates the scalar tie enumeration, weighted-survival implementation checks, stable-noise calculations, exchangeability enumeration, label-blind coordinate check, bound curves and complete-rank comparison. It exports 25 scalar cases, 200 weighted-survival cases, 60 stable-noise simulation cells, the exact rank permutations, 420 bound-grid points and the source data for Figure S1.

The complete-rank calculation in SI H.4 includes same-class pairs. Its denominators differ from the cross-class concordance analysis and are kept in a separate output table.

## 9. Population inference simulation

```sh
python reproduce.py --stage inference
```

`08_inference/inference_design.json` defines 72 binary and eight survival scenarios, each with 2,000 datasets. The default stage recomputes all coverage summaries from the 160,000 stored estimates and standard errors, regenerates datasets 0, 999 and 1999 in every scenario, repeats one 999-resample bootstrap calculation in each of eight paired scenarios, and independently checks the reference integrals.

An unavailable normal interval counts as noncoverage in unconditional coverage. Coverage conditional on positive standard error is reported separately. The standard-error ratio is the mean estimated standard error divided by the empirical standard deviation across datasets. The paired percentile comparisons use the same first 1,000 datasets per selected scenario.

To regenerate all 160,000 datasets and the eight sets of 1,000 percentile intervals, run the following commands. Each percentile interval uses 999 resamples. The reference values use 16 scrambled Sobol blocks of size 2^20 for each survival configuration. This is the full simulation run, which takes substantially longer than reaggregating the supplied arrays.

```sh
python code/inference_coverage.py --out generated/inference_full --replicates 2000 --bootstrap-replicates 1000 --bootstrap-resamples 999 --truth-power 20 --truth-blocks 16 --workers 4
python code/verify_inference.py --full-replay generated/inference_full
```

## 10. Nonlinear component perturbations

```sh
python reproduce.py --stage nonlinear
```

`b_robustness/parameters.json` specifies the fold grid, signs, multipliers and numerical tolerance. For every draw and panel, the stage evaluates h_i = s t abs(g_i - c), with 202 folds per multiplier and multipliers 1, 2, 4 and 8. There are 24,240 folds overall. `verify.py` independently recalculates the endpoints and maximizing witnesses using scalar pair loops.

At t = 1, the fold maximum exceeds the scalar maximum on 2 of 10 DILImap draws and 4 of 10 within-source draws. On within-source draw 9, both scalar endpoints move 4/96, a fold moves 6/96, and B is 8/96. Table S8 is regenerated from the same records.

## 11. Published tables, figures and numerical targets

```sh
python reproduce.py --stage report
```

Run this after the preceding stages. `render_results.py` generates 16 numerical tables and their CSV rows. `verify_reported.py` checks all 285 rows and 87 key results against the manuscript's numerical targets in `reported/`. `render_figures.py` uses these regenerated outputs to draw Figures 1, 2 and S1. Times New Roman is used when installed; otherwise the script selects an available serif font. The selected font and all plotted values are recorded in `generated/plot_verification.json`.

## Parameter replay and optional refitting

`--deep` recalculates component scores and objectives for 60 application models, three real-data gradient-validation fits and 31 frozen-reference models from their parameters and feature inputs. It also reconstructs frozen reference assets and repeats the 15 small-problem autograd checks stated in SI G.4.

The following commands perform fresh fitting into separate directories:

```sh
python data/evidence/05_reconstruction/code/run_reconstructed.py --output generated/retrained_main
python data/evidence/05_reconstruction/code/analyse_reconstructed.py --results generated/retrained_main --output-dir generated/retrained_main_summary
python data/evidence/07_reference_study/code/run_study.py --output generated/retrained_frozen
python code/rerun_fitted_simulation.py --output generated/retrained_simulation
```

The application fits were made with PyTorch 2.5.1 and float64 calculations on CUDA. The fitted-model simulation uses CPU fitting. The supplied states reproduce the published scores independently of rerunning the optimizer. Fresh fitting should record its device and software versions alongside the generated parameters.

The real-data gradient check in SI G.4 can be repeated on CUDA with:

```sh
python data/evidence/05_reconstruction/code/check_application_fits.py --output generated/application_fit_checks
```

It independently fits the three coordinate designs on the supplied four-compound draw and compares all fitted parameters and objectives with the saved record.
