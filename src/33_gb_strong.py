from sklearn.ensemble import GradientBoostingClassifier
from strong_features import make_features
import pandas as pd

X, y, X_test, ids = make_features()

model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId":ids,"Survived":preds}) \
  .to_csv("submissions/submission_v33_gb_strong.csv", index=False)
