import numpy as np
import pandas as pd


data = pd.read_csv('../data/data-processed.csv')
features = list(data.columns)[:-1]

X = data[features]
X_compressed = (X-1) / 9
X_centered = X_compressed - X_compressed.mean(axis=0)

covariance = X_centered.cov()
_, vectors = np.linalg.eig(covariance)

def restore(arr):
    return (np.linalg.inv(vectors).dot(arr) + X_compressed.mean(axis=0).values.reshape(-1, 1)) * 9