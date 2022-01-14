import numpy as np
import pandas as pd

# Load the original data
data = pd.read_csv('../data/data-processed.csv')
features = list(data.columns)[:-1]

# Rescale it to the (0, 1) interval
X = data[features]
X_compressed = (X-1) / 9
X_centered = X_compressed - X_compressed.mean(axis=0)

# Calculate the covariance and eigenvectors
covariance = X_centered.cov()
_, vectors = np.linalg.eig(covariance)

# Define functions to convert data matrices to and from PCA space
mu = X_compressed.mean(axis=0).values.reshape(1, -1)

def transform(arr):
    # Converts a vector from the original space into PCA space
    rescale = ((arr - 1) / 9) - mu
    return rescale.dot(vectors)

def recover(arr):
    # Converts a vector from PCA space to the original space
    invrs = arr.dot(np.linalg.inv(vectors))
    return (((invrs + mu) * 9) + 1)