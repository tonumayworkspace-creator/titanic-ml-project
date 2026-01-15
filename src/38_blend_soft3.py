import pandas as pd

a = pd.read_csv("submissions/submission_v32_extratrees_strong.csv")
b = pd.read_csv("submissions/submission_v33_gb_strong.csv")
c = pd.read_csv("submissions/submission_v35_logistic_strong.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"] + c["Survived"]) >= 2).astype(int)

blend.to_csv("submissions/submission_v38_blend3.csv", index=False)
