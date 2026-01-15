import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Save PassengerId for submission
test_ids = test["PassengerId"]

# Combine for preprocessing
full = pd.concat([train, test], ignore_index=True)

# Handle missing values
full["Age"] = full["Age"].fillna(full["Age"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# Features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
full = full[features + ["Survived"]]

# Encoding
full = pd.get_dummies(full, columns=["Sex", "Embarked"], drop_first=True)

# Split back
train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop(columns=["Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

# Train final model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predict test
preds = model.predict(test_processed)

# Create submission
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": preds
})

# Save file
submission.to_csv("submissions/submission_v1.csv", index=False)

print("Submission file saved to submissions/submission_v1.csv")
print(submission.head())
