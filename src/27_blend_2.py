import pandas as pd

a = pd.read_csv("submissions/submission_v11_deck_ticket.csv")
b = pd.read_csv("submissions/submission_v14_stacking.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"]) >= 1).astype(int)

blend.to_csv("submissions/submission_v22_blend2.csv", index=False)
