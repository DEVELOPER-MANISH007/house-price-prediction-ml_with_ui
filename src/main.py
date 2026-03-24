import argparse
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# 🔥 Base path (important)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "05_housing.csv")
MODEL_FILE = os.path.join(BASE_DIR, "..", "models", "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "..", "models", "pipeline.pkl")


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])


# 🔥 TRAIN FUNCTION
def train():
    print("Training started...")

    housing = pd.read_csv(DATA_PATH)

    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
        labels=[1,2,3,4,5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing['median_house_value']
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("✅ Model trained and saved!")


# 🔥 PREDICT FUNCTION
def predict():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found! Run: python src/main.py train")
        return

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    print("\nEnter details:")

    longitude = float(input("Longitude: "))
    latitude = float(input("Latitude: "))
    housing_median_age = float(input("Housing Median Age: "))
    total_rooms = float(input("Total Rooms: "))
    total_bedrooms = float(input("Total Bedrooms: "))
    population = float(input("Population: "))
    households = float(input("Households: "))
    median_income = float(input("Median Income: "))
    ocean_proximity = input("Ocean Proximity: ")

    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    transformed = pipeline.transform(input_data)
    prediction = model.predict(transformed)

    print(f"\n💰 Predicted House Price: {prediction[0]}")


# 🔥 MAIN FUNCTION
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])

    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        predict()


if __name__ == "__main__":
    main()