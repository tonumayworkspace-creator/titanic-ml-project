import pandas as pd

a = pd.read_csv("submissions/submission_v31_rf_strong.csv")
b = pd.read_csv("submissions/submission_v32_extratrees_strong.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"]) >= 1).astype(int)

blend.to_csv("submissions/submission_v36_blend1.csv", index=False)
