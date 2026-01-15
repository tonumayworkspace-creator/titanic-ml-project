import pandas as pd

a = pd.read_csv("submissions/submission_v11_deck_ticket.csv")
b = pd.read_csv("submissions/submission_v9_logistic.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"]) >= 1).astype(int)

blend.to_csv("submissions/submission_v21_blend1.csv", index=False)
