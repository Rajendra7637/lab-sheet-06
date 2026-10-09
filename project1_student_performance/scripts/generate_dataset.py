"""Generate the sample dataset for Project 1: Student Performance Prediction.

Run once:  python scripts/generate_dataset.py
Creates:   datasets/student_performance.csv

The data is made up (synthetic). It has realistic problems on purpose: a few
missing values and duplicate rows, so the preprocessing step has real work to do.
You can replace it with a real dataset (for example the UCI "Student Performance"
data from Kaggle). Keep or rename the column names, then update the column lists
at the top of the main file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
n = 500

ability = rng.normal(0, 1, n)  # hidden "ability" that links the columns together

previous_sem_result = (65 + 10 * ability + rng.normal(0, 5, n)).clip(35, 98).round(1)
attendance_pct = (76 + 6 * ability + rng.normal(0, 9, n)).clip(40, 100).round(1)
study_hours_per_day = (3.5 + 0.8 * ability + rng.normal(0, 1.3, n)).clip(0.5, 9).round(1)
internal_marks = (18 + 4 * ability + rng.normal(0, 3, n)).clip(0, 30).round(1)       # out of 30
assignment_score = (12 + 2.5 * ability + rng.normal(0, 3, n)).clip(0, 20).round(1)   # out of 20
extra_classes = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])

final_marks = (10
               + 0.30 * previous_sem_result
               + 0.12 * attendance_pct
               + 0.80 * internal_marks
               + 0.60 * assignment_score
               + 1.50 * study_hours_per_day
               + np.where(extra_classes == "Yes", 2.0, 0.0)
               + rng.normal(0, 4, n)).clip(5, 100).round(1)

df = pd.DataFrame({
    "student_id": np.arange(2001, 2001 + n),
    "attendance_pct": attendance_pct,
    "internal_marks": internal_marks,
    "assignment_score": assignment_score,
    "study_hours_per_day": study_hours_per_day,
    "previous_sem_result": previous_sem_result,
    "extra_classes": extra_classes,
    "final_marks": final_marks,
})

# Missing values (about 3-4% in some columns)
for col, k in [("attendance_pct", 18), ("study_hours_per_day", 15),
               ("assignment_score", 12), ("extra_classes", 10)]:
    df.loc[rng.choice(n, k, replace=False), col] = np.nan

# Duplicate rows
df = pd.concat([df, df.sample(6, random_state=2)], ignore_index=True)

out = Path(__file__).resolve().parent.parent / "datasets" / "student_performance.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {out}  shape={df.shape}")
