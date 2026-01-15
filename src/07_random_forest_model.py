import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["PassengerId"]

# Combine
full = pd.concat([train, test], ignore_index=True)

# -------------------------
# Feature Engineering
# -------------------------

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dr": "Rare",
    "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare"
}
full["Title"] = full["Title"].replace(title_map)

# Family features
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Fill missing
full["Age"] = full.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Select features
features = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone"
]

full = full[features + ["Survived"]]

# Encoding
full = pd.get_dummies(full, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Split back
train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# -------------------------
# Random Forest
# -------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    min_samples_split=4,
    random_state=42
)

# Cross validation score
cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("CV Accuracy:", round(cv.mean(), 4))

# Train final model
model.fit(X, y)

# Predict
preds = model.predict(test_processed).astype(int)

# Save submission
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": preds
})

submission.to_csv("submissions/submission_v3_rf.csv", index=False)

print("Saved: submissions/submission_v3_rf.csv")
