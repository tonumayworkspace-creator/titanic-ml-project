import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

# Combine
full = pd.concat([train, test], ignore_index=True)

# Basic preprocessing
full["Sex"] = full["Sex"].map({"male": 0, "female": 1})
full["Age"] = full["Age"].fillna(full["Age"].median())
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

features = ["Pclass", "Sex", "Age", "Fare"]
full = full[features + ["Survived"]]

# Split
train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]
X_test = test_df.copy()

# Proper ML pipeline (this removes convergence issues)
pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        solver="saga",
        max_iter=5000,
        n_jobs=-1
    ))
])

# Train
pipeline.fit(X, y)

# Predict
preds = pipeline.predict(X_test).astype(int)

# Save
pd.DataFrame({
    "PassengerId": ids,
    "Survived": preds
}).to_csv("submissions/submission_v17_log_poly.csv", index=False)

print("Saved: submissions/submission_v17_log_poly.csv")
