import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["PassengerId"]
full = pd.concat([train, test], ignore_index=True)

# ---------------------------
# Feature Engineering (clean, proven)
# ---------------------------

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
full["Title"] = full["Title"].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dr": "Rare",
    "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare"
})

# Family
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Missing values
full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Select features
features = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone"
]

full = full[features + ["Survived"]]

# Encode
full = pd.get_dummies(full, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Split
train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# ---------------------------
# Gradient Boosting
# ---------------------------

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("CV Accuracy:", scores.mean())

# Train final
model.fit(X, y)

# Predict
preds = model.predict(test_processed).astype(int)

# Save
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": preds
})

submission.to_csv("submissions/submission_v5_gb.csv", index=False)
print("Saved: submissions/submission_v5_gb.csv")
