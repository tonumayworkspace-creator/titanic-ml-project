from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from features import prepare_data

X, y, X_test, ids = prepare_data()

model = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=7,
    random_state=42
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
  .to_csv("submissions/submission_v7_extratrees.csv", index=False)

print("Saved v7")
