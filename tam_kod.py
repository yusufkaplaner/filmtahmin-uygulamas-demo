import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD
# =========================
df = pd.read_csv("hepsiemlak.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# =========================
# CLEAN
# =========================
df["price"] = df["list-view-price"].str.replace(".", "", regex=False)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df["rooms"] = df["celly"].str.replace(" ", "")
df["rooms"] = df["rooms"].replace("Stüdyo", "1")
df["rooms"] = df["rooms"].str.split("+").str[0]
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")

df["m2"] = df["celly 2"].str.replace(" m²", "", regex=False)
df["m2"] = pd.to_numeric(df["m2"], errors="coerce")

df["region"] = df["celly 3"]

df = df.dropna(subset=["price", "m2", "region"])

df["price"] = df["price"].astype(int)
df["m2"] = df["m2"].astype(int)

# =========================
# OUTLIER REMOVE
# =========================
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["price"] >= Q1 - 1.5 * IQR) &
        (df["price"] <= Q3 + 1.5 * IQR)]

# =========================
# FEATURE ENGINEERING
# =========================

# Nonlinear features
df["m2_squared"] = df["m2"] ** 2
df["log_m2"] = np.log1p(df["m2"])

# Region encoding
le = LabelEncoder()
df["region_encoded"] = le.fit_transform(df["region"])

# Target transform
df["log_price"] = np.log1p(df["price"])

features = [
    "m2",
    "m2_squared",
    "log_m2",
    "region_encoded"
]

X = df[features]
y = df["log_price"]

# =========================
# TRAIN TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=800,
    max_depth=15,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
actual = np.expm1(y_test)

mae = mean_absolute_error(actual, pred)
r2 = r2_score(actual, pred)

print("MAE:", round(mae, 2))
print("R2:", round(r2, 4))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("Cross Val R2:", round(cv_scores.mean(), 4))

# =========================
# SAVE
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")