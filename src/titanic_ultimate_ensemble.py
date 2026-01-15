import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ======================
# Load data
# ======================
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test], ignore_index=True)

# ======================
# Feature Engineering
# ======================

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
full["Title"] = full["Title"].replace({
    "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
    "Lady":"Rare","Countess":"Rare","Capt":"Rare","Col":"Rare",
    "Don":"Rare","Dr":"Rare","Major":"Rare","Rev":"Rare","Sir":"Rare",
    "Jonkheer":"Rare","Dona":"Rare"
})

# Family features
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Cabin deck
full["Deck"] = full["Cabin"].astype(str).str[0].replace("n", "U")

# Ticket frequency
full["TicketFreq"] = full.groupby("Ticket")["Ticket"].transform("count")

# Missing values
full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna("S")

# Binning
full["AgeBin"] = pd.cut(full["Age"], bins=[0,12,18,30,45,60,80], labels=False)
full["FareBin"] = pd.qcut(full["Fare"], 5, labels=False)

# Manual interaction features
full["Pclass_Sex"] = full["Pclass"].astype(str) + "_" + full["Sex"]
full["Embarked_Pclass"] = full["Embarked"] + "_" + full["Pclass"].astype(str)

features = [
    "Pclass","Sex","Age","Fare","Embarked",
    "Title","Deck","FamilySize","IsAlone",
    "TicketFreq","AgeBin","FareBin",
    "Pclass_Sex","Embarked_Pclass"
]

full = full[features + ["Survived"]]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"].astype(int)
X_test = test_df.copy()

# For CatBoost
cat_features = X.select_dtypes(include="object").columns.tolist()

# For LGB/XGB
X_encoded = pd.get_dummies(X)
X_test_encoded = pd.get_dummies(X_test)
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# ======================
# Optuna tuning for CatBoost
# ======================
def cat_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1200),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "Logloss",
        "verbose": False,
        "random_seed": 42
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = []

    for tr, val in cv.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_features)
        preds = model.predict(X.iloc[val]).astype(int)
        scores.append(accuracy_score(y.iloc[val], preds))

    return np.mean(scores)

print("Running Optuna (CatBoost)...")
study = optuna.create_study(direction="maximize")
study.optimize(cat_objective, n_trials=50)
best_cat_params = study.best_params
print("Best CatBoost:", best_cat_params)

# ======================
# Train 15-seed ensembles
# ======================
seeds = [2,7,11,19,23,31,37,41,43,47,53,59,61,67,71]

cat_probs = []
lgb_probs = []
xgb_probs = []

for seed in seeds:
    # CatBoost
    cat = CatBoostClassifier(**best_cat_params, random_seed=seed, verbose=False)
    cat.fit(X, y, cat_features=cat_features)
    cat_probs.append(cat.predict_proba(X_test)[:,1])

    # LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=800, random_state=seed)
    lgbm.fit(X_encoded, y)
    lgb_probs.append(lgbm.predict_proba(X_test_encoded)[:,1])

    # XGBoost
    xgbm = xgb.XGBClassifier(n_estimators=800, random_state=seed, eval_metric="logloss")
    xgbm.fit(X_encoded, y)
    xgb_probs.append(xgbm.predict_proba(X_test_encoded)[:,1])

# ======================
# Blend probabilities
# ======================
cat_avg = np.mean(cat_probs, axis=0)
lgb_avg = np.mean(lgb_probs, axis=0)
xgb_avg = np.mean(xgb_probs, axis=0)

final_probs = 0.45*cat_avg + 0.30*lgb_avg + 0.25*xgb_avg

# ======================
# Threshold optimization (OOF)
# ======================
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))

for tr, val in cv.split(X, y):
    model = CatBoostClassifier(**best_cat_params, random_seed=42, verbose=False)
    model.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_features)
    oof[val] = model.predict_proba(X.iloc[val])[:,1]

best_t, best_acc = 0.5, 0
for t in np.arange(0.3, 0.7, 0.005):
    preds = (oof >= t).astype(int)
    acc = accuracy_score(y, preds)
    if acc > best_acc:
        best_acc = acc
        best_t = t

print("Best threshold:", best_t, "CV:", best_acc)

# ======================
# Final predictions
# ======================
final_preds = (final_probs >= best_t).astype(int)

pd.DataFrame({
    "PassengerId": ids,
    "Survived": final_preds
}).to_csv("submissions/submission_ultimate.csv", index=False)

print("Saved: submissions/submission_ultimate.csv")
