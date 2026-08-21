# Importing libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Loading the dataset
df = pd.read_csv("realtor-data.zip.csv.zip")

print("Original Dataset Shape:", df.shape)
print("Dataset Columns:")
print(df.columns.tolist())


# Selecting required features
features = [
    "bed",
    "bath",
    "house_size",
    "acre_lot",
    "city",
    "state",
    "zip_code",
    "status",
    "prev_sold_date"
]

target = "price"

df = df[features + [target]]

print("Selected Dataset Shape:", df.shape)


# Cleaning the dataset
df = df.dropna(subset=["price"])

df = df[df["price"] > 0]
df = df[df["house_size"] > 0]
df = df[df["bed"] > 0]
df = df[df["bath"] > 0]


# Converting previous sold date
df["prev_sold_date"] = pd.to_datetime(
    df["prev_sold_date"],
    errors="coerce"
)


# Extracting sold year and sold month
df["sold_year"] = df["prev_sold_date"].dt.year
df["sold_month"] = df["prev_sold_date"].dt.month

df = df.drop(columns=["prev_sold_date"])

print("After Cleaning:", df.shape)


# Removing extreme prices
lower = df["price"].quantile(0.01)
upper = df["price"].quantile(0.99)

df = df[
    (df["price"] >= lower) &
    (df["price"] <= upper)
]

print("After Removing Extreme Prices:", df.shape)


# Limiting dataset size
if len(df) > 200000:
    df = df.sample(
        n=200000,
        random_state=42
    )

print("Final Dataset Shape:", df.shape)


# Creating input features
features = [
    "bed",
    "bath",
    "house_size",
    "acre_lot",
    "city",
    "state",
    "zip_code",
    "status",
    "sold_year",
    "sold_month"
]

X = df[features]


# Log transforming target variable
y = np.log1p(df["price"])

print("X Shape:", X.shape)
print("Y Shape:", y.shape)


# Defining numerical features
numeric_features = [
    "bed",
    "bath",
    "house_size",
    "acre_lot",
    "zip_code",
    "sold_year",
    "sold_month"
]


# Defining categorical features
categorical_features = [
    "city",
    "state",
    "status"
]


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

print("Numerical Preprocessing Ready!")


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

print("Categorical Preprocessing Ready!")


# Creating preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing Ready!")


# Creating Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.04,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    loss="huber",
    random_state=42
)

print("Model Created!")


# Creating complete machine learning pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

print("Pipeline Created!")


# Splitting dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Training the model
print("====================================")
print("       TRAINING STARTED")
print("====================================")

pipeline.fit(X_train, y_train)

print("Training Completed!")


# Making predictions
pred_log = pipeline.predict(X_test)

y_pred = np.expm1(pred_log)
y_actual = np.expm1(y_test)

print("Prediction Completed!")


# Evaluating model performance
mae = mean_absolute_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_pred
    )
)

r2 = r2_score(
    y_actual,
    y_pred
)


# Displaying model performance
print("====================================")
print("       MODEL PERFORMANCE")
print("====================================")

print(f"MAE          : {mae:,.2f}")
print(f"RMSE         : {rmse:,.2f}")
print(f"R2 Score     : {r2:.4f}")
print(f"R2 Percentage: {r2 * 100:.2f}%")


# Sample house prediction
sample = pd.DataFrame({
    "bed": [3],
    "bath": [2],
    "house_size": [1800],
    "acre_lot": [0.25],
    "city": ["Houston"],
    "state": ["Texas"],
    "zip_code": [77001],
    "status": ["for_sale"],
    "sold_year": [2020],
    "sold_month": [6]
})


# Predicting sample house price
prediction_log = pipeline.predict(sample)

prediction = np.expm1(prediction_log)[0]


# Displaying sample prediction
print("====================================")
print("       SAMPLE PREDICTION")
print("====================================")

print(f"Predicted House Price: ${prediction:,.2f}")


# Saving the trained model
joblib.dump(
    pipeline,
    "house_price_model.pkl"
)

print("====================================")
print("Model saved successfully!")
print("====================================")
