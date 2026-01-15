import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test])
full["SexBin"] = (full["Sex"] == "female").astype(int)
full["Child"] = (full["Age"] < 12).astype(int)

full["Age"] = full["Age"].fillna(full["Age"].median())
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

features = ["Pclass","SexBin","Child","Age","Fare"]
full = full[features + ["Survived"]]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

preds = model.predict(test_df).astype(int)
pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
    .to_csv("submissions/submission_v20_rules_ml.csv", index=False)
