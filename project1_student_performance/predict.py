
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

# (question, column name, minimum, maximum, example value)
QUESTIONS = [
    ("Attendance percentage (0-100)", "attendance_pct", 0, 100, 80),
    ("Internal marks (0-30)", "internal_marks", 0, 30, 20),
    ("Assignment score (0-20)", "assignment_score", 0, 20, 14),
    ("Study hours per day (0-24)", "study_hours_per_day", 0, 24, 3.5),
    ("Previous semester result in percent (0-100)", "previous_sem_result", 0, 100, 65),
]


def ask_number(question, minimum, maximum, example):
    """Ask for a number and keep asking until it is valid."""
    while True:
        try:
            text = input(f"{question} [{example}]: ").strip()
        except EOFError:
            return example
        if text == "":
            return example
        try:
            value = float(text)
        except ValueError:
            print("  Please type a number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"  The value must be between {minimum} and {maximum}.")


def main():
    try:
        marks_model = joblib.load(MODEL_DIR / "student_marks_model.joblib")
        level_model = joblib.load(MODEL_DIR / "student_level_model.joblib")
    except FileNotFoundError:
        raise SystemExit("Saved models not found. Run  python student_performance.py  first.")

    print("=== Student Performance Prediction ===")
    values = {column: ask_number(question, low, high, example)
              for question, column, low, high, example in QUESTIONS}
    try:
        extra = input("Attends extra classes? (Yes/No) [No]: ").strip().title() or "No"
    except EOFError:
        extra = "No"
    values["extra_classes"] = extra if extra in ("Yes", "No") else "No"

    student = pd.DataFrame([values])
    predicted_marks = marks_model.predict(student)[0]
    predicted_level = level_model.predict(student)[0]

    print("\n--- Result ---")
    print(f"Predicted final marks : {predicted_marks:.1f} / 100")
    print(f"Predicted level       : {predicted_level}")
    if predicted_level == "At Risk":
        print("Advice: this student needs extra support (attendance, assignments, study time).")


if __name__ == "__main__":
    main()
