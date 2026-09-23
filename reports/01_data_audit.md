# Step 1 - Raw Data Audit

## Overview
- Encounters: 101,766
- Unique patients: 71,518
- Columns: 50
- Patients with >1 encounter: 16,773

## Target distribution (`readmitted`)
| Value | Count | % |
|---|---|---|
| NO | 54,864 | 53.9% |
| >30 | 35,545 | 34.9% |
| <30 | 11,357 | 11.2% |

Binary target = `<30`: positive rate **11.2%** (imbalanced).

## Missing values
| Column | % missing |
|---|---|
| weight | 96.9% |
| medical_specialty | 49.1% |
| payer_code | 39.6% |
| race | 2.2% |
| diag_3 | 1.4% |
| diag_2 | 0.4% |
| diag_1 | 0.0% |

## Constant / near-constant columns
- Constant: ['examide', 'citoglipton']
- >99.9% single value: ['chlorpropamide', 'acetohexamide', 'tolbutamide', 'miglitol', 'troglitazone', 'tolazamide', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

## Clinical exclusions to handle in Step 2
- Expired/hospice discharges: 2,423 encounters (2.4%); 30-day readmission rate among them: 1.8%
- `gender` = Unknown/Invalid: 3

## Identifier columns (must never be model features)
- `encounter_id`, `patient_nbr`: IDs. `patient_nbr` is needed for a
  patient-level train/test split (Step 4), and per-patient aggregates
  are a leakage risk (Step 3).
