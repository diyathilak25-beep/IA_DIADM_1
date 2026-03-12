"""
data_generation.py
──────────────────
Generates a synthetic hiring dataset for Talabat UAE Bias Analytics.
Saves: dataset.csv  (3 000 rows × 25 columns)

Usage:
    python data_generation.py
"""

import pandas as pd
import numpy as np


def generate_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Categorical pools ─────────────────────────────────────────────────────
    genders = rng.choice(["Male", "Female"], n, p=[0.58, 0.42])

    nationalities = rng.choice(
        ["Emirati", "Indian", "Pakistani", "Filipino",
         "Egyptian", "British", "American", "Other"],
        n, p=[0.10, 0.30, 0.15, 0.10, 0.12, 0.06, 0.05, 0.12],
    )

    edu_levels = rng.choice(
        ["High School", "Bachelor's", "Master's", "PhD"],
        n, p=[0.08, 0.55, 0.30, 0.07],
    )

    fields = rng.choice(
        ["Computer Science", "Business", "Engineering", "Marketing",
         "Finance", "Hospitality", "Logistics", "Other"],
        n, p=[0.18, 0.18, 0.15, 0.10, 0.12, 0.07, 0.10, 0.10],
    )

    visa_status = rng.choice(
        ["UAE Resident", "Work Visa", "Visit Visa", "Citizen"],
        n, p=[0.50, 0.25, 0.10, 0.15],
    )

    app_source = rng.choice(
        ["LinkedIn", "Referral", "Company Website", "Job Portal", "Walk-in"],
        n, p=[0.30, 0.20, 0.20, 0.20, 0.10],
    )

    prev_tier = rng.choice(
        ["Tier 1", "Tier 2", "Tier 3", "Startup", "None"],
        n, p=[0.15, 0.25, 0.25, 0.15, 0.20],
    )

    eng_prof = rng.choice(
        ["Native", "Fluent", "Intermediate", "Basic"],
        n, p=[0.15, 0.40, 0.35, 0.10],
    )

    # ── Numeric features ──────────────────────────────────────────────────────
    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    edu_num = np.array([edu_map[e] for e in edu_levels])

    age           = rng.normal(30, 6, n).clip(20, 55).astype(int)
    exp           = rng.normal(5, 3.5, n).clip(0, 20).astype(int)
    tech_score    = rng.normal(65, 15, n).clip(20, 100).round(1)
    comm_score    = rng.normal(68, 12, n).clip(20, 100).round(1)
    int_score     = rng.normal(67, 13, n).clip(20, 100).round(1)
    assess_score  = rng.normal(65, 14, n).clip(20, 100).round(1)
    uni_rank      = rng.normal(300, 200, n).clip(1, 1000).astype(int)
    certs         = rng.integers(0, 8, n)
    salary_exp    = rng.normal(12000, 4500, n).clip(3000, 40000).round(-2).astype(int)
    hr_rating     = rng.normal(3.2, 0.7, n).clip(1, 5).round(1)
    mgr_rating    = rng.normal(3.1, 0.8, n).clip(1, 5).round(1)

    relocate      = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    leadership    = rng.choice(["Yes", "No"], n, p=[0.40, 0.60])
    internship    = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    referral      = (app_source == "Referral").astype(int)

    # ── Hiring probability with embedded bias ─────────────────────────────────
    logit = (
          0.030 * (int_score   - 50)
        + 0.020 * (tech_score  - 50)
        + 0.020 * (comm_score  - 50)
        + 0.250 * exp
        + 0.400 * (edu_num - 2)
        + 0.500 * referral                         # referral bias  ← intentional
        - 0.005 * (uni_rank - 300)                 # prestige bias  ← intentional
        + 0.600 * (hr_rating  - 3)
        + 0.500 * (mgr_rating - 3)
    )

    # Gender bias in Computer Science / Engineering
    logit += np.where(
        np.isin(fields, ["Computer Science", "Engineering"]) & (genders == "Male"),
        0.35, 0,
    )

    # Nationality advantage for Emirati / Western candidates
    logit += np.where(
        np.isin(nationalities, ["Emirati", "British", "American"]),
        0.25, 0,
    )

    prob  = 1 / (1 + np.exp(-logit / 5))
    hired = (rng.random(n) < prob).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "Candidate_ID":            [f"TLB{str(i + 1).zfill(4)}" for i in range(n)],
        "Age":                     age,
        "Gender":                  genders,
        "Nationality":             nationalities,
        "Education_Level":         edu_levels,
        "University_Ranking":      uni_rank,
        "Field_of_Study":          fields,
        "Years_of_Experience":     exp,
        "Technical_Skills_Score":  tech_score,
        "Communication_Score":     comm_score,
        "Interview_Score":         int_score,
        "Assessment_Test_Score":   assess_score,
        "Internship_Experience":   internship,
        "Previous_Company_Tier":   prev_tier,
        "Expected_Salary":         salary_exp,
        "Willing_to_Relocate":     relocate,
        "Visa_Status":             visa_status,
        "English_Proficiency":     eng_prof,
        "Leadership_Experience":   leadership,
        "Certifications_Count":    certs,
        "Application_Source":      app_source,
        "Referral_Status":         np.where(referral, "Referred", "Not Referred"),
        "HR_Rating":               hr_rating,
        "Manager_Rating":          mgr_rating,
        "Final_Hiring_Decision":   np.where(hired == 1, "Hired", "Rejected"),
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("dataset.csv", index=False)
    hired = (df["Final_Hiring_Decision"] == "Hired").sum()
    print(f"✅  dataset.csv saved — {len(df):,} rows | "
          f"{hired:,} hired ({hired/len(df)*100:.1f}%) | "
          f"{df.shape[1]} columns")
