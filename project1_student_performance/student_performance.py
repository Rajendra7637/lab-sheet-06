
# **Problem:** Predict a student's academic performance from attendance, internal
# marks, assignments, study hours and the previous semester result.
#
# We solve it in two ways:
# 1. **Regression**: predict the exact final marks (0-100).
# 2. **Classification**: predict the performance level (At Risk / Average / Excellent).
#
# **Workflow:** load data -> explore -> clean -> split -> build pipelines ->
# compare algorithms -> evaluate -> visualize -> save the best model -> predict.

# %% [markdown]
# ## Step 1: Imports and settings

# %%
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
    if BASE_DIR.name == "notebooks":
        BASE_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "datasets" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 200)          # show wide tables fully
pd.set_option("display.max_columns", 20)

RANDOM_STATE = 42
NUMERIC_COLUMNS = ["attendance_pct", "internal_marks", "assignment_score",
                   "study_hours_per_day", "previous_sem_result"]
CATEGORICAL_COLUMNS = ["extra_classes"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_REGRESSION = "final_marks"
LEVEL_NAMES = ["At Risk", "Average", "Excellent"]     # classes for classification


def save_and_show(file_name):
    """Save the current figure into the outputs folder and display it."""
    plt.savefig(OUT_DIR / file_name, dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## Step 2: Load the dataset

# %%
try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw_df = pd.read_csv(DATA_PATH)
except (FileNotFoundError, pd.errors.ParserError) as err:
    print("Error while loading dataset:", err)
    raise

print("Shape:", raw_df.shape)
print(raw_df.head())

# %% [markdown]
# ## Step 3: Explore the data (EDA)
# We look at the data types, summary statistics, missing values and duplicates.

# %%
raw_df.info()
print("\nSummary statistics:\n", raw_df.describe().round(2))
print("\nMissing values:\n", raw_df.isnull().sum())
print("\nDuplicate rows:", raw_df.duplicated().sum())

# %% [markdown]
# ## Step 4: Clean the data
# - Remove duplicate rows.
# - Check that values are in a valid range (for example attendance between 0 and 100).
# - Missing values are NOT filled here. They are filled inside the pipeline (Step 7),
#   using only the training data, so no information leaks from the test data.

# %%
clean_df = raw_df.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(clean_df))

valid_ranges = {"attendance_pct": (0, 100), "internal_marks": (0, 30), "assignment_score": (0, 20),
                "study_hours_per_day": (0, 24), "previous_sem_result": (0, 100), "final_marks": (0, 100)}
for column, (low, high) in valid_ranges.items():
    outside = ((clean_df[column] < low) | (clean_df[column] > high)).sum()
    print(f"{column}: {outside} values outside {low}-{high}")

# Create the classification target from the final marks
clean_df["performance_level"] = pd.cut(
    clean_df[TARGET_REGRESSION], bins=[-np.inf, 45, 70, np.inf], labels=LEVEL_NAMES)
print("\nStudents in each performance level:")
print(clean_df["performance_level"].value_counts().reindex(LEVEL_NAMES))

# %% [markdown]
# ## Step 5: Visualize the data

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sns.histplot(clean_df[TARGET_REGRESSION], bins=25, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of final marks")
sns.countplot(x="performance_level", data=clean_df, order=LEVEL_NAMES, ax=axes[0, 1])
axes[0, 1].set_title("Students per performance level")
sns.scatterplot(x="study_hours_per_day", y=TARGET_REGRESSION, hue="performance_level",
                hue_order=LEVEL_NAMES, data=clean_df, ax=axes[0, 2])
axes[0, 2].set_title("Study hours vs final marks")
sns.scatterplot(x="attendance_pct", y=TARGET_REGRESSION, data=clean_df, ax=axes[1, 0], alpha=0.6)
axes[1, 0].set_title("Attendance vs final marks")
sns.boxplot(x="extra_classes", y=TARGET_REGRESSION, data=clean_df, ax=axes[1, 1])
axes[1, 1].set_title("Extra classes vs final marks")
sns.heatmap(clean_df[NUMERIC_COLUMNS + [TARGET_REGRESSION]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=axes[1, 2], annot_kws={"size": 7})
axes[1, 2].set_title("Correlation heatmap")
plt.tight_layout()
save_and_show("p1_eda.png")

# %% [markdown]
# ## Step 6: Split into training and testing sets
# 80% of the students train the model, 20% are kept hidden to test it.
# For classification we use `stratify` so each level has the same share in both sets.

# %%
X = clean_df[FEATURE_COLUMNS]
y_marks = clean_df[TARGET_REGRESSION]
y_level = clean_df["performance_level"].astype(str)

X_train, X_test, y_marks_train, y_marks_test, y_level_train, y_level_test = train_test_split(
    X, y_marks, y_level, test_size=0.2, random_state=RANDOM_STATE, stratify=y_level)
print("Training students:", len(X_train), "| Testing students:", len(X_test))

# %% [markdown]
# ## Step 7: Preprocessing pipeline
# - Numeric columns: fill missing values with the **median**, then **standardize**.
# - Category column (`extra_classes`): fill missing values with the **most common** value,
#   then **one-hot encode**.
#
# The pipeline learns these values from the training data only.

# %%
numeric_pipeline = Pipeline([("fill_missing", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
categorical_pipeline = Pipeline([("fill_missing", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("numeric", numeric_pipeline, NUMERIC_COLUMNS),
                                  ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)])
print(preprocessor)

# %% [markdown]
# # Part A: Regression (predict the final marks)

# %% [markdown]
# ## Step 8: Compare regression algorithms

# %%
regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

regression_pipelines, regression_rows = {}, []
for name, model in regression_models.items():
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_marks_train)
    predictions = pipeline.predict(X_test)
    cv_r2 = cross_val_score(pipeline, X_train, y_marks_train, cv=5, scoring="r2")
    mse = mean_squared_error(y_marks_test, predictions)
    regression_pipelines[name] = pipeline
    regression_rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_marks_test, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 (test)": r2_score(y_marks_test, predictions),
        "R2 (5-fold CV)": cv_r2.mean(),
    })

regression_table = pd.DataFrame(regression_rows).set_index("Model").round(4)
print(regression_table.sort_values("R2 (test)", ascending=False))

best_regression_name = regression_table["R2 (test)"].idxmax()
best_regression = regression_pipelines[best_regression_name]
print("\nBest regression model:", best_regression_name)

# %% [markdown]
# ## Step 9: Interpret the regression metrics
# - **MAE**: the average mistake in marks.
# - **RMSE**: like MAE but punishes big mistakes more. Same unit as marks.
# - **R2**: share of the variation in marks that the model explains (1 = perfect).
# - **CV R2** close to **test R2** means the model is not just lucky on one split.

# %%
best_row = regression_table.loc[best_regression_name]
print(f"{best_regression_name}: on average the predicted marks are about "
      f"{best_row['MAE']:.1f} marks away from the real marks (MAE).")
print(f"The typical error (RMSE) is {best_row['RMSE']:.1f} marks.")
print(f"The model explains {best_row['R2 (test)'] * 100:.1f}% of the differences between students (R2).")

# %% [markdown]
# ## Step 10: Visualize the regression results

# %%
best_predictions = best_regression.predict(X_test)
residuals = y_marks_test - best_predictions

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
regression_table["R2 (test)"].sort_values().plot.barh(ax=axes[0], color="steelblue")
axes[0].set_title("R2 score of each model")
axes[0].set_xlabel("R2 (test)")

axes[1].scatter(y_marks_test, best_predictions, alpha=0.6)
axes[1].plot([y_marks_test.min(), y_marks_test.max()], [y_marks_test.min(), y_marks_test.max()], "r--")
axes[1].set_title(f"Actual vs predicted ({best_regression_name})")
axes[1].set_xlabel("Actual marks")
axes[1].set_ylabel("Predicted marks")

sns.histplot(residuals, bins=20, kde=True, ax=axes[2])
axes[2].set_title("Prediction errors (actual - predicted)")
plt.tight_layout()
save_and_show("p1_regression_results.png")

# %% [markdown]
# ## Step 11: Which factors matter most?
# Permutation importance: shuffle one column and see how much the score drops.

# %%
importance = permutation_importance(best_regression, X_test, y_marks_test, n_repeats=15,
                                    random_state=RANDOM_STATE, scoring="r2")
importance_series = pd.Series(importance.importances_mean, index=FEATURE_COLUMNS).sort_values()
print(importance_series.round(4).sort_values(ascending=False))

plt.figure(figsize=(7, 4))
importance_series.plot.barh(color="darkorange")
plt.title("Importance of each factor (drop in R2 when shuffled)")
plt.xlabel("Importance")
save_and_show("p1_feature_importance.png")

# %% [markdown]
# # Part B: Classification (predict the performance level)

# %% [markdown]
# ## Step 12: Compare classification algorithms

# %%
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

classification_pipelines, classification_rows = {}, []
for name, model in classification_models.items():
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_level_train)
    predictions = pipeline.predict(X_test)
    cv_accuracy = cross_val_score(pipeline, X_train, y_level_train, cv=5, scoring="accuracy")
    classification_pipelines[name] = pipeline
    classification_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_level_test, predictions),
        "Precision": precision_score(y_level_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_level_test, predictions, average="weighted", zero_division=0),
        "F1-score": f1_score(y_level_test, predictions, average="weighted", zero_division=0),
        "Accuracy (5-fold CV)": cv_accuracy.mean(),
    })

classification_table = pd.DataFrame(classification_rows).set_index("Model").round(4)
print(classification_table.sort_values("Accuracy", ascending=False))

best_classifier_name = classification_table["Accuracy"].idxmax()
best_classifier = classification_pipelines[best_classifier_name]
print("\nBest classification model:", best_classifier_name)

# %% [markdown]
# ## Step 13: Evaluate the best classifier
# - **Accuracy**: share of students put in the correct level.
# - **Precision**: of the students predicted as a level, how many really are in it.
# - **Recall**: of the students really in a level, how many the model found.
# - The **confusion matrix** shows which levels get mixed up.

# %%
best_classifier_predictions = best_classifier.predict(X_test)
print(classification_report(y_level_test, best_classifier_predictions, labels=LEVEL_NAMES, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ConfusionMatrixDisplay.from_predictions(y_level_test, best_classifier_predictions,
                                        labels=LEVEL_NAMES, cmap="Blues", ax=axes[0])
axes[0].set_title(f"Confusion matrix ({best_classifier_name})")
classification_table[["Accuracy", "F1-score", "Accuracy (5-fold CV)"]].plot.bar(ax=axes[1])
axes[1].set_title("Classification models compared")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
save_and_show("p1_classification_results.png")

at_risk_recall = recall_score(y_level_test, best_classifier_predictions, labels=["At Risk"],
                              average=None, zero_division=0)[0]
print(f"The model finds {at_risk_recall:.0%} of the students who are really 'At Risk'.")
print("(For a teacher, catching At Risk students early is the most useful result.)")

# %% [markdown]
# ## Step 14: Save the trained models

# %%
try:
    joblib.dump(best_regression, MODEL_DIR / "student_marks_model.joblib")
    joblib.dump(best_classifier, MODEL_DIR / "student_level_model.joblib")
    model_info = {
        "regression_model": best_regression_name,
        "regression_metrics": regression_table.loc[best_regression_name].to_dict(),
        "classification_model": best_classifier_name,
        "classification_metrics": classification_table.loc[best_classifier_name].to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "levels": LEVEL_NAMES,
    }
    with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as info_file:
        json.dump(model_info, info_file, indent=2)
    print("Saved in the models folder:")
    for saved_file in sorted(MODEL_DIR.glob("*")):
        print(" -", saved_file.name)
except OSError as err:
    print("Could not save model:", err)

# %% [markdown]
# ## Step 15: Load the saved models and predict for new students

# %%
marks_model = joblib.load(MODEL_DIR / "student_marks_model.joblib")
level_model = joblib.load(MODEL_DIR / "student_level_model.joblib")

new_students = pd.DataFrame({
    "attendance_pct": [92, 70, 55],
    "internal_marks": [27, 18, 9],
    "assignment_score": [18, 12, 6],
    "study_hours_per_day": [5.0, 3.0, 1.0],
    "previous_sem_result": [88, 65, 48],
    "extra_classes": ["Yes", "No", "No"],
})
result = new_students.copy()
result["Predicted final marks"] = marks_model.predict(new_students).round(1)
result["Predicted level"] = level_model.predict(new_students)
print(result.to_string(index=False))

# %% [markdown]
# ## Conclusion
# Several algorithms were compared for both regression and classification. The
# best models were saved with Joblib and used to predict the performance of new
# students. The most important factors are shown in the feature importance chart.
