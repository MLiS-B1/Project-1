# -*- coding: utf-8 -*-
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

# %% [markdown]
# # K-means

# %% [markdown]
# K-means is an algorithm which attempts to group each point in the dataset into exactly one of $K$ clusters based on the distance from that point to the cluster. Once each point is assigned, the position of the clusters center is updated based on the mean position of all the points assigned to it.
#
# This is a turbulent algorithm which quickly converges to local minima, however can be very informative in determining the optimum number of clusters to assign to data. This number is found at the "elbow point" of a graph plotting the model inertia agains $K$.
#
# Here we implement K-means and allow for the specification of an arbitry measure of distance between two points. The two chosen to analyse here are Euclidean and Manhattan distance.

# %% [markdown] tags=[]
# ## Data handling

# %%
# %matplotlib widget

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from itertools import repeat

import seaborn as sns

from tqdm.notebook import tqdm

import ipywidgets as widgets

# %%
np.set_printoptions(precision=2)

# %% tags=[]
orig_data = pd.read_csv(
    "../data/data-processed.csv"
)
orig_data


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Implementation of K-Means

# %%
# Utility class to store the output of the algorithm
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
# Vectorised function to assign points to clusters
def closest_cluster(row, centers=None, distance_function=None):
    dists = dict(zip(range(len(centers)), map(distance_function, centers, repeat(row.values))))
    return min(dists, key=lambda x: dists[x])


# %%
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
# Function to take the average over several models with randomised starting points
def test_models(data, cluster_range, n_epochs, distance_function="euclidean", verbose=True):
    np.random.seed(42)
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
    
    fig, ax = plt.subplots()
    
    ax.plot(cluster_range, [i.inertia for i in models], marker="o")
    
    ax.set_xlabel("Number of centroids (K)")
    ax.set_ylabel("Mean inertia")
    ax.set_xticks(cluster_range)
    
    plt.show()
    
    return models


# %% [markdown]
# ## Application to data

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Using the original data

# %%
orig_model = KMeansModel(copy(orig_data), 2)

# %%
orig_model.summary()

# %% [markdown] tags=[]
# #### Using Euclidean distance

# %%
cluster_range = range(1, 9)
epochs = 10
verbose = False

# %%
test_models(orig_data, cluster_range, epochs, distance_function="euclidean", verbose=verbose); None

# %% [markdown]
# A clear elbow point is visible at $k=2$

# %% [markdown] tags=[]
# #### Manhattan distance

# %%
test_models(orig_data, cluster_range, epochs, distance_function="manhattan", verbose=verbose); None

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### Using PCA data

# %%
pca_data = pd.read_csv("../data/data-pca.csv")

# %%
pca_model = KMeansModel(copy(pca_data), K=2, distance="euclidean")

# %% [markdown] tags=[]
# #### Using Euclidean distance

# %%
test_models(pca_data, cluster_range, epochs, distance_function="euclidean", verbose=verbose); None

# %% [markdown] tags=[]
# #### Manhattan distance

# %%
test_models(pca_data, cluster_range, epochs, distance_function="manhattan", verbose=verbose); None

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Position of the centers

# %% [markdown]
# We can now look at the position of the centers we get using these different datasets and algorithms. To make the comparison between PCA and original data meaningful, we transform the PCA center positions back into the original coordinate system.

# %%
# Custom defined functions to implement switching to and from the original dataset
from utility import transform, recover

# Pad the other dimensions with zero (the mean value for those features in PCA space)
centers_padded = np.pad(pca_model.centers, ((0, 0), (0, 6)))

# Transfer the PCA means into the space of the original data
pca_centers = recover(centers_padded)

# Construct a data frame so we can see what is happening
centers = pd.DataFrame(
    pca_centers, 
    columns=orig_data.columns[:-1],
    index=["PCA centers"] * pca_model.centers.shape[0]
)
centers = centers.append(pd.DataFrame(
        orig_model.centers,
        columns=orig_data.columns[:-1],
        index=["Original data centers"] * orig_model.centers.shape[0]
    )
)

centers

# %%
np.sqrt(np.mean((orig_model.centers - pca_centers[:, :]) ** 2))


# %% [markdown]
# RMSE between original and PCA clusters

# %% [markdown] tags=[]
# ### Visualising the data

# %% [markdown]
# We can use the PCA dataset to visualise the results of our data in a meaninfgul way

# %%
def interactive_demo(dataset, K, fig, ax, distance="euclidean", colour="class"):
    
    fig, ax = plt.subplots(figsize=plt.figaspect(0.5))

    
    if dataset == "original":
        data = orig_data
    elif dataset == "PCA":
        data = pca_data
    
    model = KMeansModel(copy(data), K=K, distance=distance)
        
    # SSC = subset class
    # SSG = subset group (cluster)
    ssc = lambda f, l: pca_data[pca_data["class"] == l][f]
    ssg = lambda f, c: pca_data[model.data["clusterIndex"] == c][f]
    
    if dataset == "original":
        model.centers = transform(model.centers)
    
    if colour == "class":
        # Create an artist for the data and clusters
        ax.scatter(ssc("PC1", 0), ssc("PC2", 0), c="orange", label="Benign", s=10)
        ax.scatter(ssc("PC1", 1), ssc("PC2", 1), c="purple", label="Malignant", s=10)
        ax.scatter(model.centers[:, 0], model.centers[:, 1], label="Cluster", marker="X", c="green", s=100)
    elif colour == "cluster":
        for i, v in enumerate(model.centers):
            ax.scatter(ssg("PC1", i), ssg("PC2", i), label=f"Cluster {i+1}", s=10)
            ax.scatter(v[0], v[1], label=f"Center {i+1}", marker="X", s=100)
    
    ax.legend()
    fig.canvas.draw()

    
widgets.interact(
    interactive_demo, 
    dataset=widgets.RadioButtons(
        options=["original", "PCA"],
        description="Which dataset to use"
    ),
    K=widgets.BoundedIntText(min=1, max=20, value=2),
    distance=widgets.RadioButtons(
        options=["euclidean", "manhattan"],
        description="Distance function"
    ),
    colour=widgets.RadioButtons(
        options=["class", "cluster"],
        description="Colour points by"
    ),
    fig=widgets.fixed(1),
    ax=widgets.fixed(1)
); None

# %%
c = orig_model.data.corr()
c.style.background_gradient(cmap="coolwarm")

# %% [markdown]
# # Specificity and sensetivity

# %%
from utility import specificty_sensetivity

# %%
pred = pca_model.data["clusterIndex"].values
truth = pca_model.data["class"].values

if np.sum(pred == truth) / len(pred) < 0.5:
    pred = np.logical_not(pred)

# %%
diff = np.stack((truth, pred), 1)

specificty_sensetivity(difference_mat=diff)

# %%
pred = orig_model.data["clusterIndex"].values

truth = orig_model.data["class"].values
diff = np.stack((truth, pred), 1)

specificty_sensetivity(difference_mat=diff)

# %% [markdown]
# Speficicity is reported first, then sensetivity then accuracy. In the case of cancer diagnosis, sensetivity is the most important factor as 
