# Step 3 - Leakage Detection

## 1. Correlation screen (|r| with 30-day readmission)
| Feature | abs(r) |
|---|---|
| **total_encounters** | 0.235 |
| number_inpatient | 0.168 |
| **prior_encounters** | 0.140 |
| total_prior_visits | 0.128 |
| number_emergency | 0.061 |
| number_diagnoses | 0.054 |
| time_in_hospital | 0.047 |
| num_medications | 0.041 |

Strongest legitimate feature: number_inpatient (0.168).

## 2. Point-in-time test
Each feature is recomputed using only encounters up to the one being
scored (what a hospital would actually know at discharge).

| Feature | Rows that change |
|---|---|
| prior_encounters | 0% |
| total_encounters | 66% |

## 3. Ablation test (patient-level split, ROC-AUC)
| Model | AUC | Change |
|---|---|---|
| baseline (no history features) | 0.676 | +0.000 |
| + prior_encounters | 0.679 | +0.003 |
| + total_encounters | 0.789 | +0.113 |

## 4. Split test (same features, no history features)
| Split | AUC |
|---|---|
| Random row split | 0.673 |
| Patient-level split | 0.676 |

The random split put 6,757 patients in both train and test.

## Verdict
- **prior_encounters: passed, kept.**
- **total_encounters: LEAKED, removed.** correlation 0.235 > 0.2; value changes for 66% of rows when future data is hidden; adds +0.113 AUC on its own.

Why `total_encounters` leaks: a patient readmitted within 30 days
necessarily has another encounter later in the data, so counting all
encounters partially encodes the answer. It is unknowable at discharge.
