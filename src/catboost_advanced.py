import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ------------------------
# Load Data
# ------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test], ignore_index=True)

# ------------------------
# Feature Engineering
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

full["Deck"] = full["Cabin"].astype(str).str[0]
full["Deck"] = full["Deck"].replace("n", "U")

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

cat_features = ["Sex", "Embarked", "Title", "Deck"]

# ------------------------
# 1. Cross-validation tuning (manual grid)
# ------------------------
params_list = [
    {"depth": 5, "learning_rate": 0.05},
    {"depth": 6, "learning_rate": 0.03},
    {"depth": 7, "learning_rate": 0.03},
]

best_score = 0
best_params = None

print("\nRunning CV tuning...")

for params in params_list:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=800,
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            loss_function="Logloss",
            verbose=False,
            random_seed=42
        )

        model.fit(X_tr, y_tr, cat_features=cat_features)
        preds = model.predict(X_val).astype(int)
        scores.append(accuracy_score(y_val, preds))

    mean_score = np.mean(scores)
    print(params, "CV Accuracy:", round(mean_score, 4))

    if mean_score > best_score:
        best_score = mean_score
        best_params = params

print("\nBest params:", best_params, "CV score:", round(best_score, 4))

# ------------------------
# 2. Train 3 models with different seeds
# ------------------------
print("\nTraining 3 models for ensemble...")

seeds = [42, 123, 2024]
test_probs = []

for seed in seeds:
    model = CatBoostClassifier(
        iterations=1500,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        loss_function="Logloss",
        verbose=False,
        random_seed=seed
    )

    model.fit(X, y, cat_features=cat_features)

    probs = model.predict_proba(X_test)[:, 1]
    test_probs.append(probs)

# ------------------------
# 3. Average probabilities
# ------------------------
avg_probs = np.mean(test_probs, axis=0)

# ------------------------
# 4. Optimize threshold using CV predictions
# ------------------------
print("\nOptimizing threshold...")

oof_probs = np.zeros(len(X))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=800,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        loss_function="Logloss",
        verbose=False,
        random_seed=42
    )

    model.fit(X_tr, y_tr, cat_features=cat_features)
    oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_acc = 0

for t in np.arange(0.3, 0.7, 0.01):
    preds = (oof_probs >= t).astype(int)
    acc = accuracy_score(y, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("Best threshold:", round(best_thresh, 3), "CV accuracy:", round(best_acc, 4))

# ------------------------
# Apply threshold to test predictions
# ------------------------
final_preds = (avg_probs >= best_thresh).astype(int)

submission = pd.DataFrame({
    "PassengerId": ids,
    "Survived": final_preds
})

submission.to_csv("submissions/submission_catboost_advanced.csv", index=False)

print("\nSaved: submissions/submission_catboost_advanced.csv")
