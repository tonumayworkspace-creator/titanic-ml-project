import pandas as pd

a = pd.read_csv("submissions/submission_v11_deck_ticket.csv")
b = pd.read_csv("submissions/submission_v10_ensemble.csv")
c = pd.read_csv("submissions/submission_v9_logistic.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"] + c["Survived"]) >= 2).astype(int)

blend.to_csv("submissions/submission_v23_blend3.csv", index=False)
