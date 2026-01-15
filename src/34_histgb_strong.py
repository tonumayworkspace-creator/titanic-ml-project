from sklearn.ensemble import HistGradientBoostingClassifier
from strong_features import make_features
import pandas as pd

X, y, X_test, ids = make_features()

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=500
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId":ids,"Survived":preds}) \
  .to_csv("submissions/submission_v34_histgb_strong.csv", index=False)
