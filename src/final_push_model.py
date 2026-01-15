import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test], ignore_index=True)

# -----------------------
# Feature engineering (high signal, low noise)
# -----------------------

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
full["Title"] = full["Title"].replace({
    "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
    "Lady":"Rare","Countess":"Rare","Capt":"Rare","Col":"Rare",
    "Don":"Rare","Dr":"Rare","Major":"Rare","Rev":"Rare","Sir":"Rare",
    "Jonkheer":"Rare","Dona":"Rare"
})

# Family size
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Ticket group size
full["TicketGroup"] = full.groupby("Ticket")["Ticket"].transform("count")

# Cabin deck
full["Deck"] = full["Cabin"].astype(str).str[0]
full["Deck"] = full["Deck"].replace("n", "U")

# Age imputation by title
full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna("S")

# Encode categoricals
full = pd.get_dummies(full, columns=["Sex","Embarked","Title","Deck"], drop_first=True)

# Features
features = [
    col for col in full.columns
    if col not in ["Survived","Name","Ticket","Cabin"]
]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()]

X = train_df[features]
y = train_df["Survived"]
X_test = test_df[features]

# -----------------------
# Model (regularized GBM)
# -----------------------
model = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=20,
    random_state=42
)

model.fit(X, y)

preds = model.predict(X_test).astype(int)

# Save
pd.DataFrame({
    "PassengerId": ids,
    "Survived": preds
}).to_csv("submissions/submission_final_push.csv", index=False)

print("Saved: submissions/submission_final_push.csv")
