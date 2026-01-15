import pandas as pd

# Load data
train = pd.read_csv("data/train.csv")

# Basic overview
print("\nShape:", train.shape)
print("\nColumns:", train.columns.tolist())

# Missing values
print("\nMissing Values:")
print(train.isnull().sum())

# Data types
print("\nData Types:")
print(train.dtypes)

# Target distribution
print("\nSurvival Rate:")
print(train["Survived"].value_counts(normalize=True))

# Basic statistics
print("\nNumerical Summary:")
print(train.describe())
