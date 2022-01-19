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

def specificty_sensetivity(difference_mat):
    # Encoding 0 for True negative, 1 for True positive, 2 for False negative, 3 for false positive
    # Define a function to mark each item as an item in the confusion matrix
    def assess(x):
        # x[0] is the true value, x1 is the predicted value
        if (x == 0).all(): return 0 # If the observations agree
        if (x == 1).all(): return 1 # we have a true prediction
        if x[1] == 0: return 2 # Else we have a false prediction
        if x[1] == 1: return 3 # either positive or negative
        raise ValueError()

    spec_sens = np.apply_along_axis(assess, 1, difference_mat)

    tn = np.sum(spec_sens == 0)
    tp = np.sum(spec_sens == 1)
    fn = np.sum(spec_sens == 2)
    fp = np.sum(spec_sens == 3)

    specificity = tn / (tn + fp) # Fraction of correctly predicted negative examples
    sensetivity = tp / (tp + fn) # Fraction of correctly predicted positive examples
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return specificity, sensetivity, accuracy