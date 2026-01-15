import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test])
full["Sex"] = full["Sex"].map({"male":0,"female":1})
full["Age"] = full["Age"].fillna(full["Age"].median())

features = ["Pclass","Sex","Age"]
full = full[features + ["Survived"]]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

model = RandomForestClassifier(n_estimators=1000, max_depth=5)
model.fit(X, y)

preds = model.predict(test_df).astype(int)
pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
    .to_csv("submissions/submission_v19_rf_minimal.csv", index=False)
