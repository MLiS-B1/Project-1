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

import ipywidgets as widgets

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
np.set_printoptions(precision=2)

class Model:
    def __init__(self, 
        inertia=None,
        centers=None,
        purities=None,
        distance_function=None,
        K=0,
        data=None
    ):
        self.inertia = inertia
        self.centers = centers
        self.purities = purities
        self.distance_function = distance_function
        self.K = K
        self.data = data
        
    def summary(self):
        print(f"K-Means model K={self.K} using {self.distance_function} distance.")
        print("Centers:")
        i = 1
        for center, purity in zip(self.centers, self.purities):
            print(f"{i} : {center} : {purity}")
            i += 1



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
    
    # Define the response model
    model = Model(K=K, distance_function=distance)
    
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
    
    # Initialise the centers
    model.centers = centers.values
    
    # Set a new column to store the cluster that each point belongs to
    data["clusterIndex"] = 0
        
    complete = False
    while not complete:

        # Assign each point to the center
        data["clusterIndex"] = data[features].apply(
            closest_cluster, 
            axis=1, 
            centers=model.centers, 
            distance_function=distance
        )

        # Update the centers position based on its points
        new_centers = np.zeros((K, len(features)))
        for i in range(K):
            cluster_points = data[data["clusterIndex"] == i]
            cluster_mean = cluster_points[features].mean().values
            new_centers[i, :] = cluster_mean

        # If the new means are equal to the previous then we have converged to a solution
        if (new_centers==model.centers).all():
            complete = True
            
        # Update the location of the centers
        model.centers = copy(new_centers)
        
    # Calculate the model inertia
    model.inertia = 0
    for point_index, point in data.iterrows():
        # Get the assigned center
        center = model.centers[int(point["clusterIndex"])]
        # Update the inertia with the squared distance
        model.inertia += distance(point[features].values, center)
    
    # Calculate the cluster purity
    model.purities = np.zeros((K, 1))
    for i, _ in enumerate(model.centers):
        subset = data[data["clusterIndex"] == i]
        mode_class = subset["class"].mode().values[0]
        subset_mode_class = subset[subset["class"] == mode_class]
        model.purities[i] = subset_mode_class.shape[0] / subset.shape[0]
        
    # Save the data used by this model so cluster assignments are known
    model.data = data
        
    
    return model

# %%
model = KMeansModel(copy(data), 2)

# %%
model.summary()

# %%
model.purities[0] == model.purities[1]


# %% [markdown]
# ## Elbow diagram generation

# %% [markdown]
# ### Euclidean distance

# %%
def test_models(data, cluster_range, n_epochs, distance_function="euclidean", verbose=True):
    models = []
    
    for i in cluster_range:
        if verbose:
            print(f"Clustering with K = {i}")
        cluster_model = Model(K=i, distance_function=distance_function)
        
        # We only count the mean of the inertia over all models.
        inertias = []
        
        it = range(n_epochs)
        if verbose:
            it = tqdm(it)
        
        for ep in it:
            model = KMeansModel(copy(data), i, distance=distance_function)
            inertias.append(model.inertia)
        
        cluster_model.inertia = np.mean(inertias)
        models.append(cluster_model)
    
    plt.plot(cluster_range, [i.inertia for i in models], marker="o")
    
    plt.xlabel("Number of centroids (K)")
    plt.ylabel("Mean inertia")
    plt.xticks(cluster_range)
    
    plt.show()
    
    return models


# %%
cluster_range = range(1, 4)
epochs = 2

# %%
models = test_models(data, cluster_range, epochs, distance_function="euclidean", verbose=True)

# %% [markdown]
# A clear elbow point is visible at $k=2$

# %% [markdown]
# ### Manhattan distance

# %%
models = test_models(data, cluster_range, epochs, distance_function="manhattan", verbose=False)

# %% [markdown]
# # Using PCA

# %%
data_pca = pd.read_csv("../data/data-pca.csv")

cols = ["PC1", "PC2", "PC3", "class"]
data_pca = data_pca[cols]

# %%
models = test_models(data_pca, cluster_range, epochs, distance_function="euclidean", verbose=False)

# %%
model = KMeansModel(copy(data_pca), K=2, distance="euclidean")


# %% tags=[]

def interactive_demo(K, distance="euclidean", colour="class"):
    model = KMeansModel(copy(data_pca), K=K, distance=distance)

    # SSC = subset class
    # SSG = subset group (cluster)
    ssc = lambda f, l: model.data[model.data["class"] == l][f]
    ssg = lambda f, c: model.data[model.data["clusterIndex"] == c][f]
    
    if colour == "class":
        # Create an artist for the data and clusters
        plt.scatter(ssc("PC1", 0), ssc("PC2", 0), c="orange", label="Benign")
        plt.scatter(ssc("PC1", 1), ssc("PC2", 1), c="purple", label="Malignant")
        plt.scatter(model.centers[:, 0], model.centers[:, 1], label="Cluster", marker="X", c="green", s=100)
    elif colour == "cluster":
        for i, v in enumerate(model.centers):
            plt.scatter(ssg("PC1", i), ssg("PC2", i), label=f"Cluster {i+1}")
            plt.scatter(v[0], v[1], label=f"Center {i+1}", marker="X", s=100)
    
    plt.legend(loc=1)
    plt.show()


# %%
widgets.interact(
    interactive_demo, 
    K=widgets.IntSlider(min=1, max=20, value=2),
    distance=widgets.RadioButtons(
        options=["euclidean", "manhattan"],
        description="Distance function"
    ),
    colour=widgets.RadioButtons(
        options=["class", "cluster"],
        description="Colour points by"
    )
)
None
