import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["PassengerId"]

full = pd.concat([train, test], ignore_index=True)

# ---------------------------
# Feature Engineering
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

# Cabin signal
full["HasCabin"] = full["Cabin"].notnull().astype(int)

# Ticket group size
full["TicketGroupSize"] = full.groupby("Ticket")["Ticket"].transform("count")

# Age by title
full["Age"] = full.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Fill remaining
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Binning
full["AgeBin"] = pd.cut(full["Age"], bins=[0, 12, 18, 35, 60, 80], labels=False)
full["FareBin"] = pd.qcut(full["Fare"], 4, labels=False)

# Select features
features = [
    "Pclass", "Sex", "Embarked", "Title",
    "FamilySize", "IsAlone",
    "HasCabin", "TicketGroupSize",
    "AgeBin", "FareBin"
]

full = full[features + ["Survived"]]

# Encoding
full = pd.get_dummies(full, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Split
train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# ---------------------------
# Tuned Random Forest
# ---------------------------

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=9,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42
)

# CV validation
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

submission.to_csv("submissions/submission_v4_advanced.csv", index=False)
print("Saved: submissions/submission_v4_advanced.csv")
