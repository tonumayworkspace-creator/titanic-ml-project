import pandas as pd

def prepare_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    test_ids = test["PassengerId"]
    full = pd.concat([train, test], ignore_index=True)

    # Title
    full["Title"] = full["Name"].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
    full["Title"] = full["Title"].replace({
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
        "Col": "Rare", "Don": "Rare", "Dr": "Rare",
        "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
        "Jonkheer": "Rare", "Dona": "Rare"
    })

    # Family features
    full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
    full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

    # Fill missing
    full["Age"] = full.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
    full["Fare"] = full["Fare"].fillna(full["Fare"].median())
    full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

    # Select features
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked",
                "Title", "FamilySize", "IsAlone"]

    full = full[features + ["Survived"]]
    full = pd.get_dummies(full, columns=["Sex", "Embarked", "Title"], drop_first=True)

    train_df = full[full["Survived"].notnull()]
    test_df = full[full["Survived"].isnull()].drop(columns=["Survived"])

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    return X, y, test_df, test_ids
