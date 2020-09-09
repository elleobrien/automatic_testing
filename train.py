import pandas as pd 
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
import pickle
import os

X, y = make_regression(10000,n_features = 10)

# Train a model
reg = Lasso().fit(X, y.ravel())
# Print out training r2
print(reg.score(X,y.ravel() ))

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.pkl'
pickle.dump(reg, open(filename, 'wb'))

