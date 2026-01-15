import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["PassengerId"]
full = pd.concat([train, test], ignore_index=True)

# ---------------------------
# Best feature set (stable)
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

# Fill missing
full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Features
features = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone"
]

full = full[features + ["Survived"]]
full = pd.get_dummies(full, columns=["Sex", "Embarked", "Title"], drop_first=True)

train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# ---------------------------
# GridSearch (professional tuning)
# ---------------------------

param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "min_samples_split": [2, 4, 6],
}

rf = RandomForestClassifier(random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X, y)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# Train final
best_model = grid.best_estimator_
best_model.fit(X, y)

preds = best_model.predict(test_processed).astype(int)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": preds
})

submission.to_csv("submissions/submission_v6_best_rf.csv", index=False)
print("Saved: submissions/submission_v6_best_rf.csv")
