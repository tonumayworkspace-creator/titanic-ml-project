import pandas as pd
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test], ignore_index=True)

# -----------------------
# Feature Engineering
# -----------------------

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
full["Title"] = full["Title"].replace({
    "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
    "Lady":"Rare","Countess":"Rare","Capt":"Rare","Col":"Rare",
    "Don":"Rare","Dr":"Rare","Major":"Rare","Rev":"Rare","Sir":"Rare",
    "Jonkheer":"Rare","Dona":"Rare"
})

# Family
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Cabin Deck
full["Deck"] = full["Cabin"].astype(str).str[0]
full["Deck"] = full["Deck"].replace("n", "U")

# Ticket frequency
full["TicketFreq"] = full.groupby("Ticket")["Ticket"].transform("count")

# Fill missing
full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna("S")

# Select features
features = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "Deck", "FamilySize", "IsAlone", "TicketFreq"
]

full = full[features + ["Survived"]]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]
X_test = test_df.copy()

# Categorical features (CatBoost magic)
cat_features = ["Sex", "Embarked", "Title", "Deck"]

# -----------------------
# Train CatBoost
# -----------------------

model = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=200
)

model.fit(X, y, cat_features=cat_features)

# Predict
preds = model.predict(X_test).astype(int)

# Save
pd.DataFrame({
    "PassengerId": ids,
    "Survived": preds
}).to_csv("submissions/submission_catboost.csv", index=False)

print("Saved: submissions/submission_catboost.csv")
