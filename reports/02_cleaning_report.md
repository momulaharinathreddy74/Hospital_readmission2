# Step 2 - Cleaning & Feature Engineering

## Row filtering log
| Step | Rows | Removed |
|---|---|---|
| Raw data | 101,766 |  |
| Exclude expired / hospice discharges | 99,343 | 2,423 |
| Exclude invalid gender | 99,340 | 3 |

Final: **99,340 encounters**, 69,987 patients, 30-day readmission rate **11.4%**.

## Dropped raw columns
- `weight`: 96.9% missing
- `medical_specialty`: replaced by grouped `specialty`
- `admission_type_id / discharge_disposition_id / admission_source_id`: replaced by UB-04 groupings
- `diag_1/2/3`: replaced by ICD-9 groups
- `readmitted`: replaced by binary `readmit_30`
- `nateglinide`: used in 0.69% of encounters (kept in med counts)
- `chlorpropamide`: used in 0.09% of encounters (kept in med counts)
- `acetohexamide`: used in 0.00% of encounters (kept in med counts)
- `tolbutamide`: used in 0.02% of encounters (kept in med counts)
- `acarbose`: used in 0.31% of encounters (kept in med counts)
- `miglitol`: used in 0.04% of encounters (kept in med counts)
- `troglitazone`: used in 0.00% of encounters (kept in med counts)
- `tolazamide`: used in 0.04% of encounters (kept in med counts)
- `examide`: used in 0.00% of encounters (kept in med counts)
- `citoglipton`: used in 0.00% of encounters (kept in med counts)
- `glyburide-metformin`: used in 0.70% of encounters (kept in med counts)
- `glipizide-metformin`: used in 0.01% of encounters (kept in med counts)
- `glimepiride-pioglitazone`: used in 0.00% of encounters (kept in med counts)
- `metformin-rosiglitazone`: used in 0.00% of encounters (kept in med counts)
- `metformin-pioglitazone`: used in 0.00% of encounters (kept in med counts)

## Features (36)
`age_years`, `gender`, `race`, `admission_type`, `discharge_group`, `admission_source`, `specialty`, `payer_code`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`, `number_outpatient`, `number_emergency`, `number_inpatient`, `total_prior_visits`, `diag_1_group`, `diag_2_group`, `diag_3_group`, `has_diabetes_primary`, `A1Cresult`, `max_glu_serum`, `a1c_tested`, `med_metformin`, `med_repaglinide`, `med_glimepiride`, `med_glipizide`, `med_glyburide`, `med_pioglitazone`, `med_rosiglitazone`, `med_insulin`, `n_meds_prescribed`, `n_meds_changed`, `med_change`, `diabetes_med`

## 30-day readmission rate by group

### discharge_group
| Group | Encounters | Readmit rate |
|---|---|---|
| Transfer_hospital | 3,483 | 18.6% |
| Facility_SNF_LTC | 17,284 | 16.0% |
| Left_AMA | 623 | 14.4% |
| Home_with_services | 13,010 | 12.7% |
| Unknown | 4,680 | 11.8% |
| Home | 60,232 | 9.3% |
| Outpatient_followup | 28 | 7.1% |

### admission_type
| Group | Encounters | Readmit rate |
|---|---|---|
| Emergency | 52,387 | 11.8% |
| Urgent | 18,132 | 11.4% |
| Unknown | 10,144 | 10.9% |
| Elective | 18,667 | 10.5% |
| Other | 10 | 10.0% |

### diag_1_group
| Group | Encounters | Readmit rate |
|---|---|---|
| Missing | 20 | 25.0% |
| Diabetes | 8,661 | 13.1% |
| Injury | 6,851 | 12.4% |
| Circulatory | 29,680 | 11.7% |
| Other | 17,793 | 11.7% |
| Genitourinary | 5,002 | 11.0% |
| Neoplasms | 3,131 | 10.9% |
| Digestive | 9,333 | 10.8% |
| Respiratory | 13,934 | 10.1% |
| Musculoskeletal | 4,935 | 9.5% |

### specialty
| Group | Encounters | Readmit rate |
|---|---|---|
| Nephrology | 1,539 | 16.1% |
| Psychiatry | 853 | 12.2% |
| Family/GeneralPractice | 7,252 | 12.1% |
| Unknown | 48,614 | 11.8% |
| InternalMedicine | 14,237 | 11.5% |
| Emergency/Trauma | 7,419 | 11.4% |
| Pulmonology | 854 | 11.2% |
| Surgery-General | 3,059 | 11.2% |
| Orthopedics | 1,392 | 10.8% |
| Urology | 682 | 9.8% |
| Other | 5,810 | 9.8% |
| Radiologist | 1,121 | 9.1% |
| Cardiology | 5,278 | 8.0% |
| Orthopedics-Reconstructive | 1,230 | 7.5% |
