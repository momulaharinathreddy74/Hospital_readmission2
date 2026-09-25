"""Every financial assumption in one place, with its source.

Change these here (or in the Step 8 dashboard); nothing else is hard-coded.
"""
ASSUMPTIONS = {
    "cost_per_readmission": {
        "value": 15_200, "low": 13_000, "high": 17_000, "unit": "$",
        "source": "AHRQ HCUP Statistical Brief #278 (2018): average cost of a "
                  "30-day all-cause adult readmission. A 2024 meta-analysis "
                  "(Healthcare 12(7):750) estimates ~$16,000-16,900.",
    },
    "effectiveness_low": {
        "value": 0.10, "unit": "%",
        "source": "Conservative end. Leppin et al., JAMA Intern Med 2014 "
                  "(42 RCTs): pooled RR 0.82, 95% CI 0.73-0.91.",
    },
    "effectiveness_high": {
        "value": 0.20, "unit": "%",
        "source": "Near the meta-analysis point estimate (18% reduction).",
    },
    "intervention_cost_per_patient": {
        "value": 500, "low": 250, "high": 1_000, "unit": "$",
        "source": "Transitional-care programs are reported at roughly $500-600 "
                  "per patient. Weakest-sourced input; varies widely by program.",
    },
}

# The dataset covers 10 years (1999-2008) across 130 hospitals, so totals over
# the whole dataset are NOT annual figures for one hospital.
DATASET_YEARS = 10


def value(name):
    return ASSUMPTIONS[name]["value"]
