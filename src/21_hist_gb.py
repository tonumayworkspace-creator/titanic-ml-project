import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test])

full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna("S")

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
full = pd.get_dummies(full[features + ["Survived"]], drop_first=True)

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)
model.fit(X, y)

preds = model.predict(test_df).astype(int)
pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
    .to_csv("submissions/submission_v16_histgb.csv", index=False)
