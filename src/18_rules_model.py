import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ids = test["PassengerId"]

full = pd.concat([train, test])

full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

# Rules:
# - Females survive
# - Boys under 10 survive
# - Adult men mostly die

def rule(row):
    if row["Sex"] == "female":
        return 1
    if row["Age"] < 10:
        return 1
    return 0

test["Survived"] = test.apply(rule, axis=1)

test[["PassengerId", "Survived"]] \
    .to_csv("submissions/submission_v13_rules.csv", index=False)

print("Saved v13")
