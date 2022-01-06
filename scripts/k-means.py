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
#     display_name: Python [conda env:mlis-project] *
#     language: python
#     name: conda-env-mlis-project-py
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from copy import copy
from itertools import repeat

import seaborn as sns

from tqdm.notebook import tqdm

# %% tags=[]
data = pd.read_csv(
    "../data/data-processed.csv"
)
data

# %% [markdown] tags=[]
# ## SKLearn implementation of K-Means

# %%
# we cannot use this this is just for understanding
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
X = data
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
inertia = []
for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
                  yaxis={'title':'Inertia'},
                 annotations=[
        dict(
            x=2,
            y=inertia[1],
            xref="x",
            yref="y",
            text="Elbow!",
            showarrow=True,
            arrowhead=7,
            ax=20,
            ay=-40
        )
    ])


# %% [markdown] tags=[]
# ## Custom implementation of K-Means
#
# - Consider the Manhattan distance metric as well as Euclidean

# %%
# Find the closest cluster for each point
def closest_cluster(row, centers=None, distance_function=None):
    dists = dict(zip(range(len(centers)), map(distance_function, centers, repeat(row.values))))
    return min(dists, key=lambda x: dists[x])

def KMeansModel(
    data, 
    K, 
    distance="euclidean"
):
    """Implementation of K-Means for a dataframe
    Step 1 - Assign all the points to the closest cluster centroid
    Step 2 - Recompute centroids of newly formed clusters
    Step 3 - Repeat until convergence"""
    
    if distance == "euclidean":
        distance = lambda x, p : np.sqrt(np.sum((x - p) ** 2)) 
    elif distance == "manhattan":
        distance = lambda x, p : np.sum(np.abs(x - p))
    else:
        raise ValueError("Argument 'distance' must be either 'euclidean' or 'manhattan'.")
                                         
    # Define which columns are features
    features = list(data.columns)[:-1]
    
    # Get K random centers (not duplicates)
    centers = data[features].sample(n=K)
    while centers.duplicated().any().any():
        centers = data[features].sample(n=K)    
    
    centers = centers.values
    
    # Set a new column to store the cluster that each point belongs to
    data["clusterIndex"] = 0
        
    complete = False
    while not complete:

        # Assign each point to the center
        data["clusterIndex"] = data[features].apply(
            closest_cluster, 
            axis=1, 
            centers=centers, 
            distance_function=distance
        )

        # Update the centers position based on its points
        new_centers = np.zeros((K, len(features)))
        for i in range(K):
            cluster_points = data[data["clusterIndex"] == i]
            cluster_mean = cluster_points[features].mean().values
            new_centers[i, :] = cluster_mean

        # If the new means are equal to the previous then we have converged to a solution
        if (new_centers==centers).all():
            complete = True
            
        # Update the location of the centers
        centers = copy(new_centers)
        
    # Calculate the model inertia
    inertia = 0
    for point_index, point in data.iterrows():
        # Get the assigned center
        center = centers[int(point["clusterIndex"])]
        # Update the inertia with the squared distance
        inertia += distance(point[features].values, center)
    
    # Calculate the cluster purity
    purities = np.zeros((K, 1))
    for i, _ in enumerate(centers):
        subset = data[data["clusterIndex"] == i]
        mode_class = subset["class"].mode().values[0]
        subset_mode_class = subset[subset["class"] == mode_class]
        purities[i] = subset_mode_class.shape[0] / subset.shape[0]
        
    
    return inertia, centers, purities

# %%
inertia, centers, purities = KMeansModel(copy(data), 2)

# %% [markdown]
# ## Elbow diagram generation

# %% [markdown]
# ### Euclidean distance

# %% tags=[]
# Cannot store the centers as the order might not be correct
inertias = []
purities = []

n_epochs = 20
cluster_range = range(1, 9)

for i in cluster_range:
    print(f"Clustering with k = {i}")
    k_inertia = []
    k_purity = []
    for ep in tqdm(range(n_epochs)):
        ep_inertia, _, ep_purity = KMeansModel(copy(data), i, distance="euclidean")
        k_inertia.append(ep_inertia)
        k_purity.append(ep_purity)
    
    mean_inertia = np.mean(k_inertia)
    mean_purity = np.mean(k_purity, axis=0)
    
    inertias.append(mean_inertia)
    purities.append(mean_purity)

# %%
plt.plot(cluster_range, inertias, marker="o")

plt.xlabel("Number of centroids")
plt.ylabel("Inertia")
plt.xticks(cluster_range)

plt.show()

# %% [markdown]
# A clear elbow point is visible at $k=2$

# %% [markdown]
# ### Manhattan distance

# %% tags=[]
# Cannot store the centers as the order might not be correct
inertias = []
purities = []

n_epochs = 20
cluster_range = range(1, 9)

for i in cluster_range:
    print(f"Clustering with k = {i}")
    k_inertia = []
    k_purity = []
    for ep in tqdm(range(n_epochs)):
        ep_inertia, _, ep_purity = KMeansModel(copy(data), i, distance="manhattan")
        k_inertia.append(ep_inertia)
        k_purity.append(ep_purity)
    
    mean_inertia = np.mean(k_inertia)
    mean_purity = np.mean(k_purity, axis=0)
    
    inertias.append(mean_inertia)
    purities.append(mean_purity)

# %%
plt.plot(cluster_range, inertias, marker="o")

plt.xlabel("Number of centroids")
plt.ylabel("Inertia")
plt.xticks(cluster_range)

plt.show()
