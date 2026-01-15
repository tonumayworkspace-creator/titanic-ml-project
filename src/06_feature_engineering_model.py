import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["PassengerId"]

# Combine
full = pd.concat([train, test], ignore_index=True)

# ---------------------------
# Feature Engineering
# ---------------------------

# 1. Title from Name
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

# Simplify rare titles
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dr": "Rare",
    "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare"
}
full["Title"] = full["Title"].replace(title_map)

# 2. Family size
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

# 3. IsAlone
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# ---------------------------
# Missing values
# ---------------------------

full["Age"] = full.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# ---------------------------
# Feature selection
# ---------------------------

features = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone"
]

full = full[features + ["Survived"]]

# ---------------------------
# Encoding
# ---------------------------

full = pd.get_dummies(full, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Split back
train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# ---------------------------
# Train model
# ---------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Predict
preds = model.predict(test_processed)

# Save submission
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": preds
})

submission.to_csv("submissions/submission_v2_fe.csv", index=False)

print("Saved: submissions/submission_v2_fe.csv")
print(submission.head())
