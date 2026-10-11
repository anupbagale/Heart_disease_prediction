# Transportability of coronary artery disease prediction models across four hospitals: a re-analysis of the UCI Heart Disease data

**Anup Bagale** · Technical report, version 2 · Code and data: https://github.com/anupbagale/Heart_disease_prediction

---

## Abstract

**Background.** The Cleveland cohort of the UCI Heart Disease dataset is a common
benchmark for machine-learning classifiers, which typically report an area under the ROC curve
(AUC) near 0.9. The dataset also contains three further cohorts that are rarely used.

**Objective.** To compare logistic regression with three flexible machine-learning models for
diagnosing angiographic coronary artery disease, and to measure how well each transfers from one
hospital to others.

**Methods.** 920 patients referred for coronary angiography in Cleveland (n = 303), Budapest
(n = 294), Zurich/Basel (n = 123) and Long Beach (n = 200). Outcome: > 50 % stenosis of any major
vessel. Models: penalised logistic regression, random forest, XGBoost and a multilayer
perceptron, all with preprocessing inside the pipeline. Internal validation by nested
cross-validation repeated 10 times on Cleveland; external validation of the Cleveland models at
the other three sites with 13 and with 10 predictors; leave-one-site-out validation.
Discrimination, calibration, decision-curve net benefit and performance by sex, with
bootstrap 95 % confidence intervals.

**Results.** On Cleveland, all models reached AUC 0.902–0.910, with no model better than logistic
regression (paired AUC differences −0.011 to +0.002; every 95 % CI included 0). At the other hospitals AUC fell
to 0.88–0.90 (Hungary), 0.75–0.78 (Switzerland) and 0.70–0.75 (VA Long Beach). Calibration was
poor wherever disease prevalence differed from Cleveland's (calibration intercept up to +3.4).
Imputing three tests that were largely unmeasured outside Cleveland made external performance worse:
at VA, sensitivity of logistic regression was 0.48 with 13 predictors and 0.85 with 10.

**Conclusions.** High internal AUC on this benchmark says little about performance at another
hospital. Flexible models offered no advantage. The predictors chosen, how their missingness is
handled and local recalibration matter more than the choice of algorithm.

---

## 1. Introduction

Clinical prediction models are often developed in one population and used in another. Their
performance usually drops, both because patient populations differ (case mix, disease prevalence)
and because measurements differ in what is recorded and how. External validation in new settings
is therefore a core requirement before clinical use (Collins et al., 2024).

The UCI Heart Disease dataset is unusual among public benchmarks because it already contains this
structure: four centres that recorded the same 13 candidate predictors for patients referred for
angiography. The original study used it to test a probability algorithm across sites
(Detrano et al., 1989). Most later machine-learning work, however, uses only the Cleveland file,
evaluates on a single random split, and reports accuracy.

This report asks three questions:

1. With a rigorous internal validation, do flexible models outperform logistic regression on Cleveland?
2. How well do Cleveland-trained models discriminate and calibrate at the other three hospitals?
3. How does structurally missing data at the new sites affect performance, and does a smaller
   predictor set transfer better?

## 2. Methods

### 2.1 Data and participants

The processed files of the UCI Heart Disease dataset (Janosi et al., 1989; CC BY 4.0) contain
consecutive patients referred for coronary angiography at the Cleveland Clinic Foundation
(n = 303), the Hungarian Institute of Cardiology, Budapest (n = 294), University Hospitals of
Zurich and Basel (n = 123) and the VA Medical Center, Long Beach (n = 200), collected in the 1980s.
No patients were excluded.

### 2.2 Outcome

The recorded diagnosis `num` ranges from 0 (< 50 % diameter narrowing in all major vessels) to
4. Following the original studies, the outcome is disease present (`num > 0`). Hungary records
only 0 and 1, so the 1–4 levels cannot be used across sites.

### 2.3 Predictors

Age, sex, chest-pain type (four categories), resting blood pressure, serum cholesterol, fasting
blood sugar > 120 mg/dl, resting ECG (three categories), maximum heart rate, exercise-induced
angina, ST depression, slope of the peak exercise ST segment (three categories), number of major
vessels coloured by fluoroscopy (0–3) and thallium stress test result (three categories). Numeric
predictors were kept continuous; categorical predictors were one-hot encoded.

A reduced set of 10 predictors omits `slope`, `ca` and `thal`, the three tests most often missing outside Cleveland.

### 2.4 Missing data

Cholesterol of 0 mg/dl (all Swiss and 49 VA patients) and one blood pressure of 0 mmHg were
recoded as missing. Missingness was minimal at Cleveland (`ca` 1.3 %, `thal` 0.7 %) but
extensive elsewhere: `ca` 96–99 %, `thal` 42–90 %, `slope` 14–65 %, cholesterol 100 % in
Switzerland, and about 26–28 % for several exercise-test variables at VA.

Missing values were imputed with the median (numeric) or the most frequent value (binary and
categorical) **of the training data**. In external validation this means values learned at the
development site. This deliberately reproduces what happens when a model is moved between
hospitals without adaptation. Better approaches are left to future work (Section 4.3).

### 2.5 Models

| Model | Hyperparameters searched |
|---|---|
| Logistic regression, L2 penalty | C ∈ {0.01, 0.1, 1, 10} |
| Random forest, 300 trees | max depth ∈ {3, 5, none}; min samples per leaf ∈ {1, 5} |
| XGBoost, 200 trees, learning rate 0.05, subsampling 0.8 | max depth ∈ {2, 3}; min child weight ∈ {1, 5} |
| Multilayer perceptron, L-BFGS | hidden layers ∈ {(16), (32, 16)}; L2 penalty α ∈ {0.1, 1, 10} |

Grids were kept small on purpose because with 303 patients large searches mainly fit noise.
Hyperparameters were chosen by 3-fold stratified cross-validation, maximising AUC.

### 2.6 Validation

* **Internal (Cleveland):** nested cross-validation. The outer loop was stratified 5-fold,
  repeated 10 times with different splits. The inner 3-fold search, imputation, scaling and
  encoding were all re-fitted inside each outer training fold.
* **External:** each model was tuned and trained on all Cleveland patients, then applied without
  change to each other hospital.
* **Leave-one-site-out:** with the 10 predictors, models were trained on three pooled hospitals
  and tested on the fourth.

### 2.7 Performance measures

* **Discrimination:** AUC.
* **Calibration:** calibration-in-the-large (intercept of a logistic model with the linear predictor as an offset; ideal 0) and calibration slope (ideal 1), with
  grouped reliability curves (Van Calster et al., 2019).
* **Overall:** Brier score.
* **Classification:** sensitivity and specificity at a probability threshold of 0.5.
* **Clinical utility:** decision-curve net benefit across thresholds 0.05–0.80 (Vickers & Elkin, 2006).
* **Uncertainty:** for internal validation, each metric was computed for each of the 10
  repeats and averaged. 95 % CIs come from 2,000 bootstrap resamples of patients, stratified by
  outcome, with the same resample applied to all repeats. Differences between models use a paired bootstrap.
* **Subgroups:** women and men separately, from the internal out-of-fold predictions.
* **Interpretation:** odds ratios from the penalised logistic regression refitted on all Cleveland data with
  reference-coded categories (1,000 bootstrap refits for CIs), and SHAP values (Lundberg & Lee, 2017) from XGBoost.

### 2.8 Software

Python 3.13, scikit-learn 1.9, XGBoost 3.4, SHAP 0.53. All seeds fixed. One command
(`python scripts/run_experiments.py`) reproduces every table and figure.

## 3. Results

### 3.1 Cohorts

| | Cleveland | Hungary | Switzerland | VA Long Beach |
|---|---|---|---|---|
| Patients | 303 | 294 | 123 | 200 |
| With disease | 139 (45.9 %) | 106 (36.1 %) | 115 (93.5 %) | 149 (74.5 %) |
| Age, mean (SD) | 54.4 (9.0) | 47.8 (7.8) | 55.3 (9.0) | 59.4 (7.8) |
| Male | 68.0 % | 72.4 % | 91.9 % | 97.0 % |
| Asymptomatic chest pain | 47.5 % | 41.8 % | 79.7 % | 65.5 % |
| Exercise-induced angina | 32.7 % | 30.4 % | 44.3 % | 64.6 % |

The hospitals differ in age, sex mix, symptom profile and, most of all, disease prevalence (36 % to 94 %).

### 3.2 Internal validation

| Model | AUC | Sensitivity | Specificity | Brier | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|
| Logistic regression | 0.908 (0.875–0.939) | 0.785 | 0.901 | 0.119 | 0.00 | 1.16 |
| Random forest | 0.906 (0.872–0.937) | 0.781 | 0.884 | 0.129 | 0.01 | 1.68 |
| XGBoost | 0.902 (0.868–0.934) | 0.796 | 0.873 | 0.124 | 0.03 | 0.85 |
| MLP | 0.910 (0.877–0.941) | 0.796 | 0.901 | 0.117 | 0.00 | 1.11 |

Paired AUC differences against logistic regression were −0.001 (−0.012 to 0.010) for random
forest, −0.006 (−0.020 to 0.009) for XGBoost and +0.002 (−0.001 to 0.005) for the MLP. With the 10
predictors, AUCs were 0.858–0.870, again with no model better than logistic regression.
The random forest's calibration slope of 1.68 (95 % CI 1.38–2.16) shows predictions that are too
close to the middle. The MLP selected the strongest L2 penalty in the grid (α = 10) in all 50 outer
folds. Because this is the edge of the grid, the data appear to favour a heavily constrained,
nearly linear network.

All models gave higher net benefit than a refer-all strategy at every threshold above about 7 %.

### 3.3 External validation

AUC (95 % CI) of Cleveland-trained models:

| Test site | Predictors | Logistic regression | Random forest | XGBoost | MLP |
|---|---|---|---|---|---|
| Hungary | 13 | 0.894 (0.851–0.930) | 0.900 | 0.896 | 0.890 |
| | 10 | 0.878 (0.830–0.917) | 0.887 | 0.880 | 0.879 |
| Switzerland | 13 | 0.748 (0.548–0.913) | 0.774 | 0.765 | 0.748 |
| | 10 | 0.758 (0.618–0.892) | 0.777 | 0.762 | 0.759 |
| VA Long Beach | 13 | 0.703 (0.618–0.780) | 0.722 | 0.719 | 0.699 |
| | 10 | 0.737 (0.651–0.813) | 0.751 | 0.751 | 0.738 |

Calibration and classification, logistic regression:

| Test site | Predictors | Mean predicted risk | Observed | Cal. intercept | Cal. slope | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|---|---|---|
| Hungary | 13 | 0.29 | 0.36 | +0.45 | 1.55 | 0.585 | 0.963 | 0.133 |
| | 10 | 0.42 | 0.36 | −0.50 | 0.91 | 0.783 | 0.819 | 0.133 |
| Switzerland | 13 | 0.51 | 0.94 | +3.21 | 0.90 | 0.522 | 0.875 | 0.272 |
| | 10 | 0.69 | 0.94 | +2.60 | 0.59 | 0.774 | 0.500 | 0.160 |
| VA Long Beach | 13 | 0.42 | 0.75 | +1.78 | 0.69 | 0.483 | 0.843 | 0.288 |
| | 10 | 0.70 | 0.75 | +0.31 | 0.57 | 0.852 | 0.471 | 0.174 |

Specificity in Switzerland rests on 8 patients and is not interpretable.
Calibration slopes below 1 at VA (0.57–0.69) indicate predictions that are too extreme for that population.

### 3.4 Leave-one-site-out

With the 10 predictors and logistic regression, AUC was 0.869 at Cleveland, 0.888 at Hungary, 0.733 at
Switzerland and 0.751 at VA. Calibration intercepts were −0.53, −0.99, +2.75 and +0.09: pooled
training over-predicted risk at the low-prevalence sites and under-predicted it in Switzerland.

### 3.5 Predictors and subgroups

The largest adjusted odds ratios were for reversible thallium defect (2.11, 1.61–2.71), number of
vessels on fluoroscopy (2.05 per SD, 1.62–2.61) and asymptomatic chest pain (2.03, 1.61–2.60), followed by
male sex, exercise angina, flat ST slope and ST depression. Maximum heart rate (0.68 per SD, 0.55–0.83) and
non-anginal chest pain (0.70, 0.56–0.89) were protective. Age, cholesterol, blood pressure, fasting
glucose and resting ECG had confidence intervals including 1. SHAP values for XGBoost ranked the same features highest.

Discrimination was similar in women (n = 97; AUC 0.915, 0.844–0.969) and men (n = 206; 0.889, 0.842–0.930).
At the 0.5 threshold, sensitivity was lower in women (0.72 vs 0.80) and specificity higher (0.97 vs 0.85),
consistent with the lower prevalence in women (26 % vs 55 %).

## 4. Discussion

### 4.1 Main findings

On Cleveland, a penalised logistic regression was as good as random forests, gradient boosting
and a neural network. This matches a broad literature on small tabular clinical datasets, where
flexible models rarely beat regression. With 303 patients and 13 predictors, there is little
non-linear structure to learn and much noise to fit.

Performance at other hospitals depended much more on the site than on the algorithm. Hungary,
whose patients were younger and less often diseased but otherwise presented similarly, retained most of the
discrimination. Switzerland and VA, almost entirely male populations in which three-quarters or more of
patients had disease, showed AUCs around 0.70–0.78. Part of this drop is expected and not a model failure: when almost everyone has disease and the case mix is narrow,
patients are harder to tell apart, and AUC falls even for a correct model.

Calibration was the larger problem. Predicted risks carried the prevalence of the
training population, so Cleveland models under-predicted at high-prevalence sites, while pooled
models over-predicted at low-prevalence ones. Updating only the intercept with a small local sample
is the standard first fix and would be required before any use.

### 4.2 Missing predictors

The most instructive result concerns missing data. At Cleveland, the vessel count and thallium
result are among the strongest predictors. Elsewhere they were rarely recorded, and imputing them from
Cleveland fills in the values of a typical *healthy* Cleveland patient (0 vessels, normal thallium).
The resulting predictions were systematically too low: at VA the mean predicted risk was 0.42
against an observed 0.75, and sensitivity was 0.48. A model that simply left these tests out lost 0.04
AUC internally but transferred far better (VA sensitivity 0.85, calibration intercept 0.31).
Internal validation cannot reveal this problem, because at Cleveland the tests are almost always present.
The practical lesson is to choose predictors that will be available where the model is used.

### 4.3 Limitations and future work

* **Sample size.** Recommendations for external validation call for at least 100 events and
  100 non-events (Collins et al., 2016). Switzerland (8 non-events) and VA (51) fall short, so their
  estimates are imprecise. At Cleveland there are about 8 events per candidate parameter, which favours penalised models.
* **Era and setting.** The data are from the 1980s and from patients already referred for
  angiography, a selected high-risk population. Diagnostic testing has changed since.
* **Missing data.** Single imputation with development-site values is a simple baseline.
  Missing-indicator variables, multiple imputation, or models that only use predictors available
  at each site could do better.
* **Uncertainty.** Bootstrap intervals reflect sampling of test patients, not the variability of model fitting.
* **Thresholds.** The 0.5 threshold is conventional. Decision curves show that the useful range depends on how clinicians weigh missed disease against unnecessary angiography.

Natural extensions are site-specific recalibration with a small number of local patients, methods
that explicitly handle site-level missingness, and federated or domain-adaptation approaches.

## 5. Conclusion

A benchmark AUC of 0.9 on the Cleveland cohort does not carry over to other hospitals. Choosing a
more flexible algorithm did not help. What mattered were the populations the models met, which predictors
those hospitals actually measure, and recalibration. Multi-site public datasets like this one are a
cheap way to test these issues before a model reaches patients.

## References

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378.
2. Collins GS, Ogundimu EO, Altman DG. Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. *Stat Med.* 2016;35(2):214–226.
3. Detrano R, Janosi A, Steinbrunn W, et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. *Am J Cardiol.* 1989;64(5):304–310.
4. Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease [Dataset]. UCI Machine Learning Repository; 1989. https://doi.org/10.24432/C52P4X
5. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems.* 2017;30.
6. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med.* 2019;17:230.
7. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making.* 2006;26(6):565–574.

---

## Appendix: TRIPOD+AI checklist (abridged)

Where each main item is addressed. "N/A" marks items that do not apply to a secondary analysis of public data.

| Item | Topic | Where / status |
|---|---|---|
| 1 | Title identifies development and validation of a prediction model | Title |
| 2 | Structured abstract | Abstract |
| 3 | Background, healthcare context, rationale | §1 |
| 4 | Objectives | §1 (three questions) |
| 5 | Data sources, dates | §2.1; `data/README.md` |
| 6 | Setting, eligibility, treatments | §2.1 (referred for angiography); treatment details not available |
| 7 | Data preparation and quality checks | §2.4; `src/heartdisease/data.py`; `tests/test_data.py` |
| 8 | Outcome definition and blinding | §2.2. Blinding of outcome assessment not reported in source data |
| 9 | Predictors, definitions, measurement | §2.3; `data/README.md` |
| 10 | Sample size | §4.3 (no a-priori calculation; adequacy discussed) |
| 11 | Missing data | §2.4, §4.2; missingness figure |
| 12 | Analytical methods: model types, tuning, internal validation | §2.5, §2.6 |
| 12 | Performance measures and uncertainty | §2.7 |
| 12 | Model updating / recalibration | Not done; discussed §4.1 |
| 13 | Class imbalance | Not resampled; prevalence shown by site; calibration assessed |
| 14 | Fairness | §3.5 (performance by sex) |
| 15 | Model output and thresholds | Predicted probability; 0.5 threshold and decision curves |
| 16 | Differences between development and evaluation data | §3.1, §2.4 |
| 17 | Ethical approval | N/A (public de-identified data) |
| 18 | Funding | None |
| 19 | Conflicts of interest | None |
| 20 | Protocol | No pre-registered protocol |
| 21 | Data availability | `data/raw/` (CC BY 4.0) |
| 22 | Code availability | This repository |
| 23 | Patient and public involvement | None |
| 24 | Participant flow and characteristics | §3.1 (no exclusions) |
| 25 | Model specification | Pipelines in `src/heartdisease/models.py`; odds ratios in `results/tables/` |
| 26 | Performance with CIs | §3.2–3.5; `results/tables/` |
| 27 | Subgroup performance | §3.5 |
| 28 | Interpretation | §4.1–4.2 |
| 29 | Limitations | §4.3 |
| 30 | Usability / implementation | Not intended for clinical use; recalibration required (§4.1) |
