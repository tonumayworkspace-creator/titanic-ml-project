import pandas as pd

a = pd.read_csv("submissions/submission_v11_deck_ticket.csv")
b = pd.read_csv("submissions/submission_v14_stacking.csv")
c = pd.read_csv("submissions/submission_v10_ensemble.csv")

blend = a.copy()
blend["Survived"] = ((a["Survived"] + b["Survived"] + c["Survived"]) == 3).astype(int)

blend.to_csv("submissions/submission_v24_blend_strict.csv", index=False)
