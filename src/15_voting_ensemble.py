from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from features import prepare_data

X, y, X_test, ids = prepare_data()

rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
lr = LogisticRegression(C=0.5, max_iter=2000)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
    voting="hard"
)

ensemble.fit(X, y)
preds = ensemble.predict(X_test).astype(int)

pd.DataFrame({"PassengerId": ids, "Survived": preds}) \
  .to_csv("submissions/submission_v10_ensemble.csv", index=False)

print("Saved v10")
