# Step 4 - Readmission Model

## Patient-level split
| Split | Encounters | Patients | Readmit rate |
|---|---|---|---|
| train | 59,613 | 41,991 | 11.4% |
| validation | 19,954 | 13,998 | 11.3% |
| test | 19,773 | 13,998 | 11.3% |

No patient appears in more than one split (asserted in code).

## Hyperparameter search (top 5 of 16, scored on validation)
| learning_rate | max_depth | max_iter | class_weight | val AUC |
|---|---|---|---|---|
| 0.03 | 5 | 200 | balanced | 0.6706 |
| 0.03 | 5 | 200 | None | 0.6691 |
| 0.03 | 5 | 400 | balanced | 0.6685 |
| 0.03 | 5 | 400 | None | 0.6681 |
| 0.03 | 3 | 200 | balanced | 0.6662 |

Chosen: `{'learning_rate': 0.03, 'max_depth': 5, 'max_iter': 200, 'class_weight': 'balanced'}`

## Results
| Model | Split | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Logistic regression | validation | 0.649 | | |
| Logistic regression | test | 0.664 | 0.214 | 0.0965 |
| Gradient boosting | validation | 0.671 | 0.215 | 0.2176 |
| **Gradient boosting** | **test** | **0.679** | 0.231 | 0.2156 |

Test ROC-AUC 0.679 is within the published range (0.64-0.7).
PR-AUC baseline (random model) = prevalence = 0.113.

## Probability sanity check (preview of Step 5)
- Mean predicted risk on test: **45.2%**
- Actual readmission rate on test: **11.3%**

## Top 10 features (permutation importance, drop in ROC-AUC)
| Feature | Importance |
|---|---|
| discharge_group | 0.0466 |
| number_inpatient | 0.0393 |
| prior_encounters | 0.0203 |
| payer_code | 0.0074 |
| diag_1_group | 0.0060 |
| total_prior_visits | 0.0056 |
| age_years | 0.0050 |
| time_in_hospital | 0.0043 |
| n_meds_prescribed | 0.0038 |
| diag_2_group | 0.0033 |