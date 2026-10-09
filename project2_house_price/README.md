# Project 2: House Price Prediction System

**Name:** <your name>  |  **Roll No:** <your roll number>
MCA 3rd Semester (2026-2027), COER University, Roorkee | Lab Sheet-06

## What it does

It predicts the price of a house (in lakh rupees) from its area, bedrooms,
bathrooms, parking spaces, age, and location.

## Files

| File | What it does |
|---|---|
| `house_price_prediction.py` | The full project (one cell per step). |
| `notebooks/house_price_prediction.ipynb` | The same project as a notebook. |
| `predict.py` | Type a house's details and get a price (for the demo). |
| `datasets/house_prices.csv` | The dataset (about 600 houses). |
| `scripts/generate_dataset.py` | Creates the dataset again if it is deleted. |
| `models/` | The saved model (made when you run the project). |
| `outputs/` | The graphs (made when you run the project). |

## Steps in the project

1. Load the data and look at it (info, statistics, missing values, duplicates).
2. Clean: remove duplicates, find outliers with the IQR rule, remove the extreme ones.
3. Draw graphs (price by location, area vs price, correlation heatmap).
4. Split into training (80%) and testing (20%) data.
5. Build a preprocessing pipeline: fill missing values, scale numbers, one-hot encode the location.
6. Compare Linear Regression, Ridge, Decision Tree, Random Forest and Gradient Boosting.
7. Tune Gradient Boosting with GridSearchCV.
8. Evaluate with MAE, MSE, RMSE, MAPE, R2, cross-validation, and a train vs test check for overfitting.
9. Find the most important features and save the best model with Joblib.

## How to run

```
python -m venv venv
venv\Scripts\activate          (Linux/Mac: source venv/bin/activate)
pip install -r ../requirements.txt
python house_price_prediction.py   # trains, evaluates and saves the model
python predict.py                  # demo: predict one house price
```

## Results

<After you run it, write your best model name and its scores here.>

## Conclusion

<Write 2-3 lines in your own words.>
