import pandas as pd

a = pd.read_csv("submissions/submission_v31_rf_strong.csv")
b = pd.read_csv("submissions/submission_v32_extratrees_strong.csv")
c = pd.read_csv("submissions/submission_v33_gb_strong.csv")
d = pd.read_csv("submissions/submission_v34_histgb_strong.csv")
e = pd.read_csv("submissions/submission_v35_logistic_strong.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"] + c["Survived"] + d["Survived"] + e["Survived"]) >= 3).astype(int)

blend.to_csv("submissions/submission_v40_blend5.csv", index=False)
