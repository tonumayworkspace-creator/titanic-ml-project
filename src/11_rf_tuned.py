from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from features import prepare_data

X, y, X_test, ids = prepare_data()

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_split=3,
    random_state=42
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
  .to_csv("submissions/submission_v6_rf_tuned.csv", index=False)

print("Saved v6")
