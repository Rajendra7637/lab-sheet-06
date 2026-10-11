# Project 2: House Price Prediction System

**Name:** Rajendra Singh Bisht  |  **Roll No:** 253028065
MCA 3rd Semester (2026-2027), COER University, Roorkee | Lab Sheet-06

## What it does

It predicts the price of a house (in lakh rupees) from its area, bedrooms,
bathrooms, parking spaces, age, and location.

## Files

| File | What it does |
|---|---|
| `house_price_prediction.ipynb` | project notebook. |
| `datasets/house_prices.csv` | The dataset (about 600 houses). |
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



