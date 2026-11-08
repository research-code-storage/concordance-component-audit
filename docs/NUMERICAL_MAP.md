# Numerical result index

Run `python reproduce.py` first. Each row below gives the published display value and the exact field in a newly generated result file. Array indices are zero-based. The full-precision targets and field arrays are in [`../reported/key_results.json`](../reported/key_results.json).

| Result identifier | Paper location | Display value | Result file | JSON field |
|---|---|---:|---|---|
| `clinical.n_patients` | Section 3.2; Table 4 | 69 | [`generated/survival_example.json`](../generated/survival_example.json) | `n_patients` |
| `clinical.n_comparable_pairs` | Section 3.2; Table 4 | 1653 | [`generated/survival_example.json`](../generated/survival_example.json) | `n_comparable_pairs` |
| `clinical.concordance_full` | Section 3.2; Table 4 | 0.6476 | [`generated/survival_example.json`](../generated/survival_example.json) | `concordance_full` |
| `clinical.delete_age_sex.concordance_retained` | Section 3.2; Table 4; SI J | 0.6385 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / concordance_retained` |
| `clinical.delete_age_sex.net_change` | Section 3.2; Table 4; SI J | 0.0091 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / net_change` |
| `clinical.delete_age_sex.mean_absolute_pair_change` | Section 3.2; Table 4; SI J | 0.3182 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / mean_absolute_pair_change` |
| `clinical.delete_age_sex.tie_aware_bound` | Section 3.2; Table 4; SI J | 0.4162 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / tie_aware_bound` |
| `clinical.delete_age_sex.strict_reversals` | Section 3.2; Table 4; SI J | 240 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / strict_reversals` |
| `clinical.delete_age_sex.tie_transitions` | Section 3.2; Table 4; SI J | 572 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / tie_transitions` |
| `clinical.delete_age_sex.changed_pairs` | Section 3.2; Table 4; SI J | 812 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / changed_pairs` |
| `clinical.delete_age_sex.reinforced_untied_pairs` | Section 3.2; Table 4; SI J | 162 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / reinforced_untied_pairs` |
| `clinical.delete_age_sex.opposed_untied_pairs` | Section 3.2; Table 4; SI J | 240 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_age_sex / opposed_untied_pairs` |
| `clinical.bootstrap.delete_age_sex.net_change_same_fit.lower` | Section 3.2; Table S13 | -0.038 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_age_sex / training_bootstrap / net_change_same_fit / lower` |
| `clinical.bootstrap.delete_age_sex.net_change_same_fit.upper` | Section 3.2; Table S13 | 0.025 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_age_sex / training_bootstrap / net_change_same_fit / upper` |
| `clinical.bootstrap.delete_age_sex.movement_same_fit.lower` | Section 3.2; Table S13 | 0.173 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_age_sex / training_bootstrap / movement_same_fit / lower` |
| `clinical.bootstrap.delete_age_sex.movement_same_fit.upper` | Section 3.2; Table S13 | 0.450 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_age_sex / training_bootstrap / movement_same_fit / upper` |
| `clinical.bootstrap.delete_age_sex.movement_refit.lower` | Section 3.2; Table S13 | 0.173 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_age_sex / training_bootstrap / movement_refit / lower` |
| `clinical.bootstrap.delete_age_sex.movement_refit.upper` | Section 3.2; Table S13 | 0.451 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_age_sex / training_bootstrap / movement_refit / upper` |
| `clinical.delete_ecog.concordance_retained` | Section 3.2; Table 4; SI J | 0.5780 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / concordance_retained` |
| `clinical.delete_ecog.net_change` | Section 3.2; Table 4; SI J | 0.0696 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / net_change` |
| `clinical.delete_ecog.mean_absolute_pair_change` | Section 3.2; Table 4; SI J | 0.1954 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / mean_absolute_pair_change` |
| `clinical.delete_ecog.tie_aware_bound` | Section 3.2; Table 4; SI J | 0.3987 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / tie_aware_bound` |
| `clinical.delete_ecog.strict_reversals` | Section 3.2; Table 4; SI J | 316 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / strict_reversals` |
| `clinical.delete_ecog.tie_transitions` | Section 3.2; Table 4; SI J | 14 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / tie_transitions` |
| `clinical.delete_ecog.changed_pairs` | Section 3.2; Table 4; SI J | 330 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / changed_pairs` |
| `clinical.delete_ecog.reinforced_untied_pairs` | Section 3.2; Table 4; SI J | 336 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / reinforced_untied_pairs` |
| `clinical.delete_ecog.opposed_untied_pairs` | Section 3.2; Table 4; SI J | 316 | [`generated/survival_example.json`](../generated/survival_example.json) | `delete_ecog / opposed_untied_pairs` |
| `clinical.bootstrap.delete_ecog.net_change_same_fit.lower` | Section 3.2; Table S13 | 0.025 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_ecog / training_bootstrap / net_change_same_fit / lower` |
| `clinical.bootstrap.delete_ecog.net_change_same_fit.upper` | Section 3.2; Table S13 | 0.150 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_ecog / training_bootstrap / net_change_same_fit / upper` |
| `clinical.bootstrap.delete_ecog.movement_same_fit.lower` | Section 3.2; Table S13 | 0.054 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_ecog / training_bootstrap / movement_same_fit / lower` |
| `clinical.bootstrap.delete_ecog.movement_same_fit.upper` | Section 3.2; Table S13 | 0.359 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_ecog / training_bootstrap / movement_same_fit / upper` |
| `clinical.bootstrap.delete_ecog.movement_refit.lower` | Section 3.2; Table S13 | 0.056 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_ecog / training_bootstrap / movement_refit / lower` |
| `clinical.bootstrap.delete_ecog.movement_refit.upper` | Section 3.2; Table S13 | 0.517 | [`generated/survival_refit.json`](../generated/survival_refit.json) | `delete_ecog / training_bootstrap / movement_refit / upper` |
| `simulation.aligned.net` | Section 3.1; Table 3 | 0.0380 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / aligned / net` |
| `simulation.aligned.moved` | Section 3.1; Table 3 | 0.1212 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / aligned / moved` |
| `simulation.aligned.bound` | Section 3.1; Table 3 | 0.2252 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / aligned / bound` |
| `simulation.aligned.hidden` | Section 3.1; Table 3 | 6.46 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / aligned / hidden` |
| `simulation.aligned.slack` | Section 3.1; Table 3 | 1.89 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / aligned / slack` |
| `simulation.aligned.ratio_of_means` | Section 3.1; Table 3 | 3.19 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / aligned / ratio_of_means` |
| `simulation.independent.net` | Section 3.1; Table 3 | 0.0282 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / independent / net` |
| `simulation.independent.moved` | Section 3.1; Table 3 | 0.1264 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / independent / moved` |
| `simulation.independent.bound` | Section 3.1; Table 3 | 0.2117 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / independent / bound` |
| `simulation.independent.hidden` | Section 3.1; Table 3 | 16.82 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / independent / hidden` |
| `simulation.independent.slack` | Section 3.1; Table 3 | 1.76 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / independent / slack` |
| `simulation.independent.ratio_of_means` | Section 3.1; Table 3 | 4.48 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / independent / ratio_of_means` |
| `simulation.opposed.net` | Section 3.1; Table 3 | 0.0798 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / opposed / net` |
| `simulation.opposed.moved` | Section 3.1; Table 3 | 0.1451 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / opposed / moved` |
| `simulation.opposed.bound` | Section 3.1; Table 3 | 0.2244 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / opposed / bound` |
| `simulation.opposed.hidden` | Section 3.1; Table 3 | 2.72 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / opposed / hidden` |
| `simulation.opposed.slack` | Section 3.1; Table 3 | 1.65 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / opposed / slack` |
| `simulation.opposed.ratio_of_means` | Section 3.1; Table 3 | 1.82 | [`generated/simulation_usefulness.json`](../generated/simulation_usefulness.json) | `by / component / opposed / ratio_of_means` |
| `toxicogenomic.PHH.auc_mean` | Section 3.3; Figure 2; Tables S4-S5 | 0.8750 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / auc_mean` |
| `toxicogenomic.PHH.budget_max` | Section 3.3; Figure 2; Tables S4-S5 | 0.0000 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / budget_max` |
| `toxicogenomic.PHH.changed_pairs_max` | Section 3.3; Figure 2; Tables S4-S5 | 0 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / changed_pairs_max` |
| `toxicogenomic.PHH.vulnerable_max` | Section 3.3; Figure 2; Tables S4-S5 | 0 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / vulnerable_max` |
| `certificate.PHH.t1` | Section 3.3; Figure 2 | 0.0000 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / budget_curve / 1` |
| `certificate.PHH.t2` | Section 3.3; Figure 2 | 0.0417 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / budget_curve / 2` |
| `certificate.PHH.t4` | Section 3.3; Figure 2 | 0.1250 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / budget_curve / 4` |
| `certificate.PHH.t8` | Section 3.3; Figure 2 | 0.2917 | [`generated/results.json`](../generated/results.json) | `60|PHH|matched_centred__softplus_original / budget_curve / 8` |
| `toxicogenomic.external.auc_mean` | Section 3.3; Figure 2; Tables S4-S5 | 0.6064 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / auc_mean` |
| `toxicogenomic.external.budget_max` | Section 3.3; Figure 2; Tables S4-S5 | 0.0076 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / budget_max` |
| `toxicogenomic.external.changed_pairs_max` | Section 3.3; Figure 2; Tables S4-S5 | 3 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / changed_pairs_max` |
| `toxicogenomic.external.vulnerable_max` | Section 3.3; Figure 2; Tables S4-S5 | 4 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / vulnerable_max` |
| `certificate.external.t1` | Section 3.3; Figure 2 | 0.0076 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / budget_curve / 1` |
| `certificate.external.t2` | Section 3.3; Figure 2 | 0.0114 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / budget_curve / 2` |
| `certificate.external.t4` | Section 3.3; Figure 2 | 0.0189 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / budget_curve / 4` |
| `certificate.external.t8` | Section 3.3; Figure 2 | 0.0379 | [`generated/results.json`](../generated/results.json) | `60|external|matched_centred__softplus_original / budget_curve / 8` |
| `toxicogenomic.source.auc_mean` | Section 3.3; Figure 2; Tables S4-S5 | 0.4292 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / auc_mean` |
| `toxicogenomic.source.budget_max` | Section 3.3; Figure 2; Tables S4-S5 | 0.0833 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / budget_max` |
| `toxicogenomic.source.changed_pairs_max` | Section 3.3; Figure 2; Tables S4-S5 | 6 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / changed_pairs_max` |
| `toxicogenomic.source.vulnerable_max` | Section 3.3; Figure 2; Tables S4-S5 | 8 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / vulnerable_max` |
| `certificate.source.t1` | Section 3.3; Figure 2 | 0.0833 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / budget_curve / 1` |
| `certificate.source.t2` | Section 3.3; Figure 2 | 0.3542 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / budget_curve / 2` |
| `certificate.source.t4` | Section 3.3; Figure 2 | 0.6667 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / budget_curve / 4` |
| `certificate.source.t8` | Section 3.3; Figure 2 | 0.7708 | [`generated/results.json`](../generated/results.json) | `60|source|matched_centred__softplus_original / budget_curve / 8` |
| `permutation.PHH.deletion_difference_mean` | Section 3.3; Table S6 | 0.0000 | [`generated/permutation_contrast.json`](../generated/permutation_contrast.json) | `PHH|matched_centred / deletion_difference_mean` |
| `permutation.PHH.permutation_importance_mean` | Section 3.3; Table S6 | 0.0082 | [`generated/permutation_contrast.json`](../generated/permutation_contrast.json) | `PHH|matched_centred / permutation_importance_mean` |
| `frozen_reference.PHH.n4` | SI H.3 | 0.7458 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `PHH / 4` |
| `frozen_reference.PHH.n20` | SI H.3 | 0.7250 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `PHH / 20` |
| `frozen_reference.PHH.n60` | SI H.3 | 0.7125 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `PHH / 60` |
| `frozen_reference.PHH.n102` | SI H.3 | 0.7083 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `PHH / 102` |
| `permutation.external.deletion_difference_mean` | Section 3.3; Table S6 | 0.0017 | [`generated/permutation_contrast.json`](../generated/permutation_contrast.json) | `external|matched_centred / deletion_difference_mean` |
| `permutation.external.permutation_importance_mean` | Section 3.3; Table S6 | 0.0015 | [`generated/permutation_contrast.json`](../generated/permutation_contrast.json) | `external|matched_centred / permutation_importance_mean` |
| `frozen_reference.external.n4` | SI H.3 | 0.6348 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `external / 4` |
| `frozen_reference.external.n20` | SI H.3 | 0.6333 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `external / 20` |
| `frozen_reference.external.n60` | SI H.3 | 0.6331 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `external / 60` |
| `frozen_reference.external.n102` | SI H.3 | 0.6326 | [`generated/frozen_reference.json`](../generated/frozen_reference.json) | `external / 102` |

## Worked clinical calculation

For deletion of age and sex, the 1,653 comparable pairs contain 240 strict reversals and 572 transitions to a tie. Thus A = (240 + 0.5 × 572) / 1653 = 0.3182093164. Net concordance change is the signed sum of the same pair contributions divided by 1653, giving 15 / 1653 = 0.0090744102. The different values reflect cancellation, not different pair sets.

The complete scalar-loop calculation is in `code/survival_example.py`. Its input is the regenerated patient score file, and its output includes both the integer pair counts and the normalized quantities.

## Table and figure coverage

Main Tables 2–4 and SI Tables S1–S13 contain 285 numeric rows. The targets in `reported/table_numeric_rows.json` use the printed row order and precision. `generated/tables/Table_Main2.csv`, `Table_Main3.csv`, `Table_Main4.csv`, and `Table_S1.csv` through `Table_S13.csv` reproduce these numeric rows. Readable headings and column definitions appear in the companion LaTeX tables and the manuscript. Main Table 1 and SI Table S14 are conceptual comparisons.

Figure 1 uses the 200-draw fitted-simulation means and Monte Carlo standard errors. Figure 2A uses compound-bootstrap AUC summaries, while Figures 2B–C use per-draw certificate curves and their maxima. Figure S1 uses the regenerated analytic/Monte Carlo curves and rank counterexamples. Their exact input records are copied to `generated/plot_data/` and hashed in `generated/plot_input_manifest.json`.
