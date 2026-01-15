import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test])

# Title
full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

# Deck from Cabin
full["Deck"] = full["Cabin"].astype(str).str[0]
full["Deck"] = full["Deck"].replace("n", "Unknown")

# Ticket prefix
full["TicketPrefix"] = full["Ticket"].str.replace(r"\d+", "", regex=True)
full["TicketPrefix"] = full["TicketPrefix"].replace("", "None")

# Fill missing
full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Family
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

features = ["Pclass", "Sex", "Age", "Fare", "Embarked",
            "Title", "Deck", "TicketPrefix", "FamilySize"]

full = full[features + ["Survived"]]
full = pd.get_dummies(full, drop_first=True)

train_df = full[full["Survived"].notnull()]
test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

model = RandomForestClassifier(n_estimators=500, max_depth=7, random_state=42)
model.fit(X, y)

preds = model.predict(test_df).astype(int)

pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
  .to_csv("submissions/submission_v11_deck_ticket.csv", index=False)

print("Saved v11")
