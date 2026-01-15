from sklearn.ensemble import ExtraTreesClassifier
from strong_features import make_features
import pandas as pd

X, y, X_test, ids = make_features()

model = ExtraTreesClassifier(
    n_estimators=1000,
    max_depth=10,
    random_state=42
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId":ids,"Survived":preds}) \
  .to_csv("submissions/submission_v32_extratrees_strong.csv", index=False)
