# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Information
#
# PCA might not be suitable for our data - see Bishop p559 - although our data may be modelled by a continuous latent variable in some sense, describing "how cancerous" the points are (and inventing the notion of cancerousness)

# %%
# %matplotlib widget

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets

# %% tags=[]
plt.style.use("seaborn-whitegrid")

# %%
data = pd.read_csv('../data/data-processed.csv')
data

# %%
features = list(data.columns)[:-1]

X = data[features]
y = data["class"]

# %% [markdown]
# # PCA
#
# The variance of the data doesn't need to be modified, as this is not normalization. However for PCA it is important for the column means to be centered around zero. Additionally, to make the scale of the data more managable the 1-10 scale will be compressed into the the interval (0, 1).

# %%
# Compress the scale of the data into the (0, 1) interval
X_compressed = (X-1) / 9

# Center each column about its mean (so mu=0)
X_centered = X_compressed - X_compressed.mean(axis=0)

# These two operations combined should give each column with range = 1
X_centered.max(axis=0) - X_centered.min(axis=0)

X_centered.shape

# %%
covariance = X_centered.cov()
covariance

# %% [markdown]
# step2: Eigen value decomposition

# %%
values, vectors = np.linalg.eig(covariance)

# %% [markdown]
# Step 3. Project the centered data onto the eigenvectors of $\Sigma$

# %%
PC_data = pd.DataFrame(
    np.dot(X_centered, vectors), 
    columns=[f"PC{i+1}" for i in range(vectors.shape[0])]
)
PC_data["class"] = data["class"]

# %% [markdown]
# We can calculate the fraction of each original feature which contributes to a given PC

# %%
contributions = np.apply_along_axis(
    lambda x: np.abs(x) / np.sum(np.abs(x)), 
    1, 
    vectors
)

# %%
contributions = pd.DataFrame(contributions, columns=features, index=PC_data.columns[:-1])
contributions.sort_values("PC1", axis=1, ascending=False, inplace=True)
contributions


# %% [markdown]
# These contributions can be visualised

# %% tags=[]
def plot_axes(data, vectors, col_names, show_data, show_vectors):
    origin = np.zeros(2)
    
    plt.figure()
    
    if show_data:
        wh = lambda f, l: data[data["class"] == l][f]
        plt.scatter(wh("PC1", 0), wh("PC2", 0), label="Benign", s=10)
        plt.scatter(wh("PC1", 1), wh("PC2", 1), label="Malignant", s=10)

    plt.scatter(*origin, label="Origin")
        
    vector_components = lambda comp: np.array([
        vectors[0, comp],vectors[1, comp]
    ])
    origin = (0, 0)
    if show_vectors:
        for i in range(vectors.shape[1]):
            plt.arrow(*origin, *vector_components(i), width=0.01)
            plt.text(*(1.1 * vector_components(i)), col_names[i], fontsize=14)

    plt.grid()

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.legend(loc=1)
    plt.show()

widgets.interact(
    plot_axes,
    data=widgets.fixed(PC_data),
    vectors=widgets.fixed(vectors),
    col_names=widgets.fixed(features),
    show_data=widgets.Checkbox(value=True),
    show_vectors=widgets.Checkbox(value=True)
)

# %%
contributions.T.plot.bar(stacked=True)


# %%
def plotpc(component):
    label = f"PC{component}"
    
    wh = lambda f, l: PC_data[PC_data["class"] == l][f]
        
    b = wh(f"PC{component}", 0)
    m = wh(f"PC{component}", 1)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(b, [0] * len(b), label="Benign")
    plt.scatter(m, [0] * len(m), label="Malignant")
    
    plt.legend(loc=1)
    plt.show()
    
widgets.interact(
    plotpc, 
    component=widgets.IntSlider(min=1, max=7, value=1)
); None

# %% [markdown]
# We can alter the size of the datapoints according to the third principal component to force an illusion of depth on the chart.

# %%
plt.figure()
sns.scatterplot(x=PC_data['PC1'], y=PC_data['PC2'], s=5 * np.exp(6 * PC_data["PC3"]), hue=PC_data['class'])
plt.show()

# %%
# Get the values for feature f which has label l
resset = lambda f, l: PC_data[PC_data["class"] == l][f]

# Create a plot and 3d axes
fig = plt.figure()
ax = plt.axes(projection="3d")

# Create an artist for each of the benign and malignant cases
ax.scatter3D(resset("PC1", 0), resset("PC2", 0), resset("PC3", 0), label="Benign")
ax.scatter3D(resset("PC1", 1), resset("PC2", 1), resset("PC3", 1), label="Malignant")

# Set the axes labels
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Display the legend
ax.legend(loc=1)

# %% [markdown]
# # Data export

# %%
PC_data.to_csv("../data/data-pca.csv", index=False)
