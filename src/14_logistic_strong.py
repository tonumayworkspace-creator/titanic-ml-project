from sklearn.linear_model import LogisticRegression
import pandas as pd
from features import prepare_data

X, y, X_test, ids = prepare_data()

model = LogisticRegression(
    C=0.3,
    solver="liblinear",
    max_iter=2000
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
  .to_csv("submissions/submission_v9_logistic.csv", index=False)

print("Saved v9")
