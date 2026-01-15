import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test])
full["Sex"] = full["Sex"].map({"male":0,"female":1})
full["Age"] = full["Age"].fillna(full["Age"].median())
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

features = ["Pclass","Sex","Age","Fare","SibSp","Parch"]
full = full[features + ["Survived"]]

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

model = ExtraTreesClassifier(n_estimators=800, max_depth=None, random_state=42)
model.fit(X, y)

preds = model.predict(test_df).astype(int)
pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
    .to_csv("submissions/submission_v18_extratrees_deep.csv", index=False)
