from sklearn.ensemble import RandomForestClassifier
from strong_features import make_features
import pandas as pd

X, y, X_test, ids = make_features()

model = RandomForestClassifier(
    n_estimators=800,
    max_depth=8,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId":ids,"Survived":preds}) \
  .to_csv("submissions/submission_v31_rf_strong.csv", index=False)
