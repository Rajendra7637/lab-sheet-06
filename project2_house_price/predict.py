"""Predict the price of ONE house with the saved model.

Run (after running house_price_prediction.py once):
    python predict.py

Type the values when asked, or press Enter to use the example value in brackets.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

# (question, column name, minimum, maximum, example value)
QUESTIONS = [
    ("Area in square feet (300-10000)", "area_sqft", 300, 10000, 1200),
    ("Number of bedrooms (1-10)", "bedrooms", 1, 10, 3),
    ("Number of bathrooms (1-10)", "bathrooms", 1, 10, 2),
    ("Parking spaces (0-5)", "parking", 0, 5, 1),
    ("Age of the property in years (0-100)", "age_years", 0, 100, 8),
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


def ask_location(locations):
    """Show the known locations and let the user pick one."""
    print("Locations:")
    for number, name in enumerate(locations, start=1):
        print(f"  {number}. {name}")
    while True:
        try:
            text = input(f"Choose a location number [1]: ").strip()
        except EOFError:
            return locations[0]
        if text == "":
            return locations[0]
        if text.isdigit() and 1 <= int(text) <= len(locations):
            return locations[int(text) - 1]
        print(f"  Please type a number from 1 to {len(locations)}.")


def main():
    try:
        model = joblib.load(MODEL_DIR / "house_price_model.joblib")
        with open(MODEL_DIR / "model_info.json", encoding="utf-8") as info_file:
            locations = json.load(info_file)["locations"]
    except FileNotFoundError:
        raise SystemExit("Saved model not found. Run  python house_price_prediction.py  first.")

    print("=== House Price Prediction ===")
    values = {column: ask_number(question, low, high, example)
              for question, column, low, high, example in QUESTIONS}
    values["location"] = ask_location(locations)

    house = pd.DataFrame([values])
    predicted_price = model.predict(house)[0]
    print("\n--- Result ---")
    print(f"Predicted price: {predicted_price:.1f} lakh rupees")


if __name__ == "__main__":
    main()
