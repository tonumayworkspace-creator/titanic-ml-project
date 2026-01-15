import pandas as pd

def make_features():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    ids = test["PassengerId"]

    full = pd.concat([train, test], ignore_index=True)

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

    # Cabin
    full["Deck"] = full["Cabin"].astype(str).str[0]
    full["Deck"] = full["Deck"].replace("n", "U")

    # Ticket frequency
    full["TicketFreq"] = full.groupby("Ticket")["Ticket"].transform("count")

    # Fill missing
    full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
    full["Fare"] = full["Fare"].fillna(full["Fare"].median())
    full["Embarked"] = full["Embarked"].fillna("S")

    # Binning (this helps trees)
    full["AgeBin"] = pd.cut(full["Age"], bins=[0,12,18,30,45,60,80], labels=False)
    full["FareBin"] = pd.qcut(full["Fare"], 5, labels=False)

    features = [
        "Pclass","Sex","Embarked","Title","Deck",
        "FamilySize","IsAlone","TicketFreq","AgeBin","FareBin"
    ]

    full = pd.get_dummies(full[features + ["Survived"]], drop_first=True)

    train_df = full[full["Survived"].notnull()]
    test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    return X, y, test_df, ids
