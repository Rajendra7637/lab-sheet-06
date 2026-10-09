# Project 1: Student Performance Prediction System

**Name:** Rajendra Singh Bisht  |  **Roll No:** 253026065
MCA 3rd Semester (2026-2027), COER University, Roorkee | Lab Sheet-06

## What it does

It predicts how a student will perform from attendance, internal marks,
assignment score, study hours, previous semester result, and extra classes.
It does this in two ways:

- **Regression:** predicts the final marks (0-100).
- **Classification:** predicts the level: At Risk, Average or Excellent.

## Files

| File | What it does |
|---|---|
| `student_performance.py` | The full project (one cell per step). |
| `notebooks/student_performance.ipynb` | The same project as a notebook. |
| `predict.py` | Type a student's details and get a prediction (for the demo). |
| `datasets/student_performance.csv` | The dataset (about 500 students). |
| `scripts/generate_dataset.py` | Creates the dataset again if it is deleted. |
| `models/` | The saved models (made when you run the project). |
| `outputs/` | The graphs (made when you run the project). |

## Steps in the project

1. Load the data and look at it (info, statistics, missing values, duplicates).
2. Clean: remove duplicates and check the value ranges.
3. Draw graphs (distribution, relationships, correlation heatmap).
4. Split into training (80%) and testing (20%) data.
5. Build a preprocessing pipeline: fill missing values, scale numbers, encode categories.
6. Compare algorithms.
   - Regression: Linear Regression, Ridge, Random Forest, Gradient Boosting.
   - Classification: Logistic Regression, Decision Tree, Random Forest.
7. Evaluate: MAE, MSE, RMSE, R2 for regression. Accuracy, precision, recall,
   F1 and the confusion matrix for classification. Also 5-fold cross-validation.
8. Find the most important factors (permutation importance).
9. Save the best models with Joblib and predict for new students.

## How to run

```
python -m venv venv
venv\Scripts\activate          (Linux/Mac: source venv/bin/activate)
pip install -r ../requirements.txt
python student_performance.py      # trains, evaluates and saves the models
python predict.py                  # demo: predict for one student
```

