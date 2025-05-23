import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("wine_quality_dataset.csv")
X = df.drop('quality', axis=1)
y = df['quality']

model = LinearRegression()
model.fit(X, y)

with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)
