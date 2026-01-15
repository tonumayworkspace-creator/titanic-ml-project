from sklearn.linear_model import LogisticRegression
from strong_features import make_features
import pandas as pd

X, y, X_test, ids = make_features()

model = LogisticRegression(max_iter=3000)
model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId":ids,"Survived":preds}) \
  .to_csv("submissions/submission_v35_logistic_strong.csv", index=False)
