import pandas as pd

a = pd.read_csv("submissions/submission_v11_deck_ticket.csv")  # best
b = pd.read_csv("submissions/submission_v14_stacking.csv")
c = pd.read_csv("submissions/submission_v9_logistic.csv")

blend = a.copy()
blend["Survived"] = ((2*a["Survived"] + b["Survived"] + c["Survived"]) >= 2).astype(int)

blend.to_csv("submissions/submission_v25_blend_weighted.csv", index=False)
