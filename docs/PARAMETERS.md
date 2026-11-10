# Parameters and numerical conventions

Paths beginning with a numbered directory below are relative to `data/evidence/`.

## Pairwise quantities

Every score is oriented so a larger value predicts the positive class or an earlier event. Binary applications use all positive-negative compound pairs with unit weights. The clinical analysis uses observed comparable pairs with unit weights. A score tie receives one half, with exact equality of float64 differences. Tolerance is used only to compare independently calculated values, never to change a score tie into a strict ordering.

`Delta` is full minus retained concordance. `A` is the mean absolute change of the pair contributions. `B` is the weighted mean of their magnitude-based worst-sign bounds, including half credit at ties. `V` is the normalized total weight of pairs with `abs(D) <= abs(E)`; flagged-pair counts are also reported. The multiplier grid is `[0, 0.25, 0.5, 1, 2, 4, 8]`. For tolerance epsilon, `t_epsilon` is the largest grid point with `B(t) <= epsilon`. Per-draw multiplier ranges and the minimum across draws are distinct summaries.

## Toxicogenomic application

The authoritative settings and evaluation identifiers are `05_reconstruction/protocol.json` and `05_reconstruction/results/experiments/config.json`.

| Setting | Value |
|---|---|
| Source holdout | 20 compounds, compound-sampling seed 12345 |
| Training | 60 compounds per draw, draw seeds 0 through 9 |
| Embeddings / pathways | 768 / 54 features |
| Main reference design | `matched_centred` |
| Other reported designs | `historical_centred` = pooled-centred; `revised_origin` = origin-preserving |
| Full predictor | `softplus_original`: Gaussian-kernel term plus two nonnegative fitted coordinate weights |
| Reference scores | `kernel_original`: separately fitted kernel-only model; `fixed_sum`: equal-weight sum of the two coordinates |
| Gaussian kernel | exp(-squared distance / 1536) |
| Loss parameter C / coordinate penalty lambda | 20 / 0.001 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Learning-rate schedule | Cosine, minimum 0.00001 |
| Maximum updates / patience | 1000 / 80 |
| Minimum objective improvement / gradient norm cap | 0.0001 / 1 |
| Parameter precision / optimizer seed | float64 / 0 |
| Full-model coordinate initialization | Raw Softplus parameter 1, giving log(1 + exp(1)) = 1.3132616877555847 |
| Prototype tails | 0.15 and 0.85 quantiles, two clusters in each tail |
| K-means | 10 starts, random state 42 |
| Compound bootstrap | 20,000 class-stratified resamples, seed 20260922 |
| Kernel-score permutation | 1,000 per draw, panel and design; base seed 20260929 |

The saved state records the actual training identifiers, scales, prototypes and fitted parameters for every draw. The complete permutation seeds are in `08_permutation/permutation_config.json`. They use the explicit design offsets in `config.json`; they do not depend on the order of columns in a score file.

Profiles are averaged within compound before calculating AUCs or pairwise quantities. Evaluation intervals resample compounds within class using shared resamples across draws and prediction scores. The mean AUC is an average of ten draw-specific AUCs, not an AUC obtained by averaging scores across draws.

The main application standardizes features using each training draw and uses profile-level class-balanced hinge weights C / (2 n_y), where n_y counts training profiles in that class. The frozen-reference sensitivity fixes reference coordinates and scales from 102 source compounds, then varies labelled training size over 4, 20, 60 and 102. There are ten draws at each smaller size and one at 102. Its profile hinge weight is C / (2 N_y m_g), where N_y is the number of training compounds in that class and m_g is the number of profiles for that compound. Full settings are in `07_reference_study/protocol.json`.

## Clinical application

The supplied `07_reference_study/data/lung.csv` has event coding 0 = censored and 1 = death. The single patient missing ECOG is excluded. NumPy `default_rng(20260928)` supplies the unstratified permutation used to split 227 patients into 158 training and 69 evaluation patients. Training means and population standard deviations are used to standardize age, sex and numeric ECOG. Cox regression uses Efron handling of tied event times. The evaluation set has 1,653 comparable pairs.

Evaluation-sample bootstrap: 5,000 patient resamples, seed 20261005, fixed fitted coefficients. Training-sample bootstrap: 1,000 patient resamples, seed 20261019, refitting the full and both reduced models in every replicate and evaluating on the fixed 69 patients. Normal intervals use 1.959963984540054 standard errors. Both interval types have nominal coverage 95%.

## Simulation designs

### Movement and slack

The 72 cells cross n in {50, 200}, strength a in {0.01, 0.1, 0.5, 1}, geometry s in {1, 0, -1}, and normal, scaled t(3) or rounded-normal retained errors. Equal numbers of cases and controls are fixed. The retained score is 0.6 Y + error, rounded for the retained-tie design. The component is a (Z + 0.6 s Y), with independent standard normal Z. Each cell has 200 datasets. Master seed 20260927 generates the 14,400 explicit replicate seeds in `07_reference_study/results/simulation/all_14400_replicates.csv`.

The movement-to-net-change ratio averages A / abs(Delta) over replicates with nonzero Delta. The bound-to-movement ratio averages B / A over replicates with positive A. The ratio of means is exported separately. Additional implementation checks of the multiplier envelope and bounded retained-score perturbations are recorded with the same datasets, as described in SI A and E.

### Population inference

`08_inference/inference_design.json` lists all 80 scenarios. The 72 binary scenarios use the preceding score designs. The eight survival scenarios cross n in {50, 200}, a in {0.1, 0.5}, and expected censoring fractions {0.3, 0.6}. The retained and component scores are X and a Z, where X and Z are independent standard normals. Event times are exponential with rate exp(X + 0.6 Z); censoring is independent exponential with the rate recorded in each scenario.

Each scenario has 2,000 datasets, generated with master seed 20261009. Bootstrap master seed 20261010 is used for 999 resamples on the first 1,000 datasets in each of eight specified scenarios. Binary resampling is within outcome class; survival resampling is by subject and reconstructs comparable pairs. Binary reference values use numerical integration. Survival reference values average 16 scrambled Sobol blocks of 2^20 points, integrating event and censoring times analytically. The reference values and integration errors are in `08_inference/reference_values.json`.

Unconditional normal-interval coverage counts a zero or invalid estimated standard error as unavailable and not covering. Positive-standard-error coverage is also exported. The standard-error ratio is mean estimated standard error divided by the empirical standard deviation across datasets.

### Fitted-model simulation

`04_results/extension_double/simulation_design.json` records 200 independent training draws of 300 observations with 30 embedding coordinates and one fixed evaluation sample of 300. Evaluation seed is 7777; training seeds are 1000 through 1199. Embedding displacements are 0, 6 and 16 along the fixed unit direction generated by the evaluation seed. Coordinate types are informative and noise. Initial coordinate weights are 0 and 1.3132616877555847. The Gaussian kernel has gamma = 1/60, C = 20 and lambda = 0.001.

Fitting uses projected nonnegative Adam updates on CPU in float64, 1000 maximum updates, learning rate 0.001 decreasing to 0.00001, patience 80, improvement threshold 0.0001 and gradient norm cap 1. Each fitted model is evaluated at all three displacements. Figure 1 shows means and two Monte Carlo standard errors across the 200 training draws.

### Nonlinear perturbations

`b_robustness/parameters.json` specifies h_i = s t abs(g_i - c), with signs -1 and 1, 101 equally spaced c values from min(g) to max(g), and multipliers 1, 2, 4 and 8. Three panels and ten draws give 24,240 folded perturbations. Equality in pair scoring remains exact; 1e-12 is the independent calculation tolerance. The finite-grid maximum is reported separately from the theoretical upper bound for the complete perturbation class.
