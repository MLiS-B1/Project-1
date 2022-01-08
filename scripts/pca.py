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

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets

# %%
data = pd.read_csv('../data/data-processed.csv')
data

# %%
features = list(data.columns)[1:-1]

### Get the features data
X = data[features]
y = data["class"]

# %% [markdown]
# # PCA
#
# Since every thing is between 1-10, there is no need to scale the data.
#
# step: 1 getting Covariance matrix

# %%
features = X.T
cov_matrix = np.cov(features)
cov_matrix[1:9]

# %% [markdown]
# step2: Eigen value decomposition

# %%
values, vectors = np.linalg.eig(cov_matrix)
values[1:9]

# %%
explained_var = []
for i in range(len(values)):
    explained_var.append(values[i] / np.sum(values))
 
# print(np.sum(explained_var), '\n', explained_var)

# %%
res = pd.DataFrame(X.dot(vectors.T[0]), columns=['PC1'])
res['PC2'] = X.dot(vectors.T[1])
res['PC3'] = X.dot(vectors.T[2])
res['PC4'] = X.dot(vectors.T[3])
res['PC5'] = X.dot(vectors.T[4])
res['PC6'] = X.dot(vectors.T[5])
res['PC7'] = X.dot(vectors.T[6])
res['Y'] = y

res


# %%
#1D plot
def plotpc(component):
    label = f"PC{component}"
    
    plt.figure(figsize=(20, 10))
    sns.scatterplot(x=res[label], y=[0] * len(res), hue=res['Y'], s=100)


# %%
widgets.interact(
    plotpc, 
    component=widgets.IntSlider(min=1, max=7, value=1)
)

# %%
#2D plot
plt.figure(figsize=(20, 10))
sns.scatterplot(x=res['PC1'], y=res['PC2'], s=10 * np.exp(res["PC3"]), hue=res['Y'])       #3D


# %%
def drawax3d(elevation, rotation, comp4scale):
    rescale_component = lambda x: np.abs(res["PC4"].min()) + 1e-4 + comp4scale * x

    
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")

    ax.scatter3D(res["PC1"], res["PC2"], res["PC3"], c=res["Y"], s=rescale_component(res["PC4"]))
    ax.view_init(elevation, rotation)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

# %%
widgets.interact(
    drawax3d, 
    elevation=widgets.IntSlider(min=0, max=90, value=30),
    rotation=widgets.IntSlider(min=0, max=360, value=30),
    comp4scale=widgets.IntSlider(min=1, max=100, value=20)
)

# %% [markdown]
# # Data export

# %%
res.to_csv("../data/data-pca.csv", index=False)
