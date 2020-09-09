import pickle
from sklearn.datasets import make_regression
import json
import pandas as pd

model = pickle.load(open("models/model.pkl", "rb"))

X_test, y = make_regression(1000,n_features = 11)

# Test on the model
y_hat = model.predict(X_test)


