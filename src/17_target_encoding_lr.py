import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

# Combine
full = pd.concat([train, test], ignore_index=True)

# -------------------------
# Feature Engineering
# -------------------------

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

# Target encoding (ONLY from training rows)
means = (
    full[full["Survived"].notnull()]
    .groupby("Title")["Survived"]
    .mean()
)

full["TitleEncoded"] = full["Title"].map(means)

# Fix unseen titles → use global mean
global_mean = full[full["Survived"].notnull()]["Survived"].mean()
full["TitleEncoded"] = full["TitleEncoded"].fillna(global_mean)

# Fill missing values
full["Age"] = full["Age"].fillna(full["Age"].median())
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna("S")

# Encode categoricals
full["Sex"] = full["Sex"].map({"male": 0, "female": 1})
full["Embarked"] = full["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# -------------------------
# Select features
# -------------------------

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "TitleEncoded"
]

full = full[features + ["Survived"]]

# Split back
train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# -------------------------
# Train model
# -------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Predict
preds = model.predict(test_df).astype(int)

# Save submission
submission = pd.DataFrame({
    "PassengerId": ids,
    "Survived": preds
})

submission.to_csv("submissions/submission_v12_target_encoding.csv", index=False)

print("Saved: submissions/submission_v12_target_encoding.csv")
