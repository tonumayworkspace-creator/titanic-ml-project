import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Combine for consistent preprocessing
full = pd.concat([train, test], ignore_index=True)

# ---------------------------
# 1. Handle missing values
# ---------------------------

# Fill Age with median
full["Age"] = full["Age"].fillna(full["Age"].median())

# Fill Embarked with most common
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Fill Fare (only in test)
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# ---------------------------
# 2. Feature selection
# ---------------------------

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
full = full[features + ["Survived"]]

# ---------------------------
# 3. Encode categorical
# ---------------------------

full = pd.get_dummies(full, columns=["Sex", "Embarked"], drop_first=True)

# ---------------------------
# 4. Split back train/test
# ---------------------------

train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Processed features:")
print(X.columns.tolist())
