import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load processed data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Preprocess again (temporary duplication for now)
full = pd.concat([train, test], ignore_index=True)

full["Age"] = full["Age"].fillna(full["Age"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
full = full[features + ["Survived"]]

full = pd.get_dummies(full, columns=["Sex", "Embarked"], drop_first=True)

train_processed = full[full["Survived"].notnull()]

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", round(acc, 4))
