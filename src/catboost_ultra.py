import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ------------------------
# Load data
# ------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test], ignore_index=True)

# ------------------------
# Feature engineering
# ------------------------
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
full["Title"] = full["Title"].replace({
    "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
    "Lady":"Rare","Countess":"Rare","Capt":"Rare","Col":"Rare",
    "Don":"Rare","Dr":"Rare","Major":"Rare","Rev":"Rare","Sir":"Rare",
    "Jonkheer":"Rare","Dona":"Rare"
})

full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

full["Deck"] = full["Cabin"].astype(str).str[0].replace("n", "U")
full["TicketFreq"] = full.groupby("Ticket")["Ticket"].transform("count")

full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna("S")

features = [
    "Pclass","Sex","Age","Fare","Embarked",
    "Title","Deck","FamilySize","IsAlone","TicketFreq"
]

full = full[features + ["Survived"]]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"].astype(int)
X_test = test_df.copy()

cat_features = ["Sex","Embarked","Title","Deck"]

# ------------------------
# Optuna tuning
# ------------------------
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 600, 1500),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "Logloss",
        "verbose": False,
        "random_seed": 42
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for tr, val in cv.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_features)
        preds = model.predict(X.iloc[val]).astype(int)
        scores.append(accuracy_score(y.iloc[val], preds))

    return np.mean(scores)

print("\nRunning Optuna optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

best_params = study.best_params
print("Best params:", best_params)

# ------------------------
# Train 5-seed ensemble
# ------------------------
seeds = [42, 7, 99, 123, 2024]
test_probs = []
oof_probs = np.zeros(len(X))

for seed in seeds:
    model = CatBoostClassifier(**best_params, random_seed=seed, verbose=False)
    model.fit(X, y, cat_features=cat_features)
    test_probs.append(model.predict_proba(X_test)[:,1])

# ------------------------
# Average probabilities
# ------------------------
avg_test_probs = np.mean(test_probs, axis=0)

# ------------------------
# Threshold optimization
# ------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(X))

for tr, val in cv.split(X, y):
    model = CatBoostClassifier(**best_params, random_seed=42, verbose=False)
    model.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_features)
    oof_probs[val] = model.predict_proba(X.iloc[val])[:,1]

best_thresh, best_acc = 0.5, 0
for t in np.arange(0.3, 0.7, 0.005):
    preds = (oof_probs >= t).astype(int)
    acc = accuracy_score(y, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("Best threshold:", best_thresh, "CV accuracy:", best_acc)

# ------------------------
# Final predictions
# ------------------------
final_preds = (avg_test_probs >= best_thresh).astype(int)

pd.DataFrame({
    "PassengerId": ids,
    "Survived": final_preds
}).to_csv("submissions/submission_catboost_ultra.csv", index=False)

print("\nSaved: submissions/submission_catboost_ultra.csv")
