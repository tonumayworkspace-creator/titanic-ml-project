from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from features import prepare_data

X, y, X_test, ids = prepare_data()

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

model.fit(X, y)
preds = model.predict(X_test).astype(int)

pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
  .to_csv("submissions/submission_v8_gb_tuned.csv", index=False)

print("Saved v8")
