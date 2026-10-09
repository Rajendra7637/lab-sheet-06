"""Generate the sample dataset for Project 2: House Price Prediction.

Run once:  python scripts/generate_dataset.py
Creates:   datasets/house_prices.csv

The data is made up (synthetic). It has realistic problems on purpose: missing
values, a few extreme houses (outliers) and duplicate rows, so the preprocessing
step has real work to do.
You can replace it with a real dataset (for example a Kaggle house price CSV).
Keep or rename the columns, then update the column lists at the top of the main file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(9)
n = 600

locations = ["City Centre", "Old Town", "New Township", "Suburb North", "Suburb South", "Near Highway"]
location_factor = {"City Centre": 1.6, "Old Town": 1.1, "New Township": 1.35,
                   "Suburb North": 0.95, "Suburb South": 0.85, "Near Highway": 0.75}

location = rng.choice(locations, n, p=[0.15, 0.15, 0.2, 0.2, 0.15, 0.15])
area_sqft = rng.lognormal(mean=7.1, sigma=0.3, size=n).clip(450, 4500).round(0)
bedrooms = np.clip(np.round(area_sqft / 520 + rng.normal(0, 0.6, n)), 1, 6).astype(int)
bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5).astype(int)
parking = np.clip(np.round(area_sqft / 1400 + rng.normal(0, 0.7, n)), 0, 3).astype(int)
age_years = rng.integers(0, 36, n)

factor = pd.Series(location).map(location_factor).to_numpy()
price_lakh = (area_sqft * 0.05 * factor
              + bedrooms * 3.0 + bathrooms * 2.5 + parking * 4.0
              - age_years * 0.45
              + rng.normal(0, 6, n)).clip(12, None).round(1)

df = pd.DataFrame({
    "house_id": np.arange(7001, 7001 + n),
    "area_sqft": area_sqft,
    "bedrooms": bedrooms.astype(float),
    "bathrooms": bathrooms.astype(float),
    "parking": parking.astype(float),
    "age_years": age_years.astype(float),
    "location": location,
    "price_lakh": price_lakh,
})

# A few extreme houses (outliers)
for index in rng.choice(n, 6, replace=False):
    df.loc[index, "area_sqft"] = float(rng.integers(6000, 9000))
    df.loc[index, "price_lakh"] = round(float(df.loc[index, "price_lakh"]) * rng.uniform(2.5, 4.0), 1)

# Missing values (about 3-4% in some columns)
for col, k in [("bathrooms", 20), ("age_years", 25), ("parking", 15), ("location", 12), ("area_sqft", 8)]:
    df.loc[rng.choice(n, k, replace=False), col] = np.nan

# Duplicate rows
df = pd.concat([df, df.sample(8, random_state=4)], ignore_index=True)

out = Path(__file__).resolve().parent.parent / "datasets" / "house_prices.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {out}  shape={df.shape}")
