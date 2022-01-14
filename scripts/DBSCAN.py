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
# # DBSCAN

# %% [markdown] tags=[]
# # TODO
#
# Use N-dimensions in the clustering process

# %%
# %matplotlib widget

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from functools import partial
import ipywidgets as widgets

import seaborn as sns

# %%
plt.style.use("seaborn-whitegrid")

# %% [markdown]
# Import the data to use for the model

# %%
data = pd.read_csv("../data/data-pca.csv")


# %% [markdown]
# ## Functions

# %%
def distance(a, b):
    return np.sqrt(np.sum((a-b) ** 2))


# %%
def mark_core_point(x, radius, min_pts):
    return np.sum(x <= radius) >= min_pts


# %%
def compute_distance_to_all_points(x, data):
    d = partial(distance, b=x)
    distances = np.apply_along_axis(d, 1, data)
    # print(distances.reshape(1, data.shape[0]).shape)
    return distances


# %% [markdown]
# ## Model

# %%
def DBSCAN(d, radius=.25, core_point_threshold=10, verbose=True):
    np.random.seed(42)
    
    n_instances, n_dims = d.shape    

    # Define an empty set of cluster indices
    cluster = np.full((n_instances), np.nan)

    # Compute the distance matrix mapping each point to eachother
    compute_distance = partial(compute_distance_to_all_points, data=d)
    distance_matrix = np.apply_along_axis(compute_distance, 1, d)

    # Calculate the core points
    mcp_partial = partial(mark_core_point, radius=radius, min_pts=core_point_threshold)
    core_points = np.apply_along_axis(mcp_partial, 1, distance_matrix)

    # Define a mask for the unassigned core points
    unassigned_core_points = np.logical_and(core_points, np.isnan(cluster))

    cluster_index = 0
    while d[unassigned_core_points].shape[0] > 0:
        # Set the random seed
        np.random.seed(42)

        # Define the index of this cluster
        cluster_index += 1

        # Check there are unassigned core points to sample
        sample_pool = np.where(np.logical_and(core_points, np.isnan(cluster)))[0]
        if len(sample_pool) == 0:
            if verbose:
                print("No remaining core points to sample")
            break
        index = np.random.choice(sample_pool)
        center = d[index]
        cluster[index] = cluster_index

        it = 0
        while True:
            it += 1
            # Neighbours is the matrix of points which lie within radius of any other point in the cluster
            neighbours = (distance_matrix[cluster == cluster_index] <= radius)
            any_neighbour = np.any(neighbours, axis=0)

            # Stop if there are no new neighbours (ie the number of points which are in the radius of the
            # cluster is equal to the number of points in the cluster)
            core_neighbours = np.logical_and(any_neighbour, core_points)
            if np.sum(any_neighbour[core_neighbours == True]) <= np.sum(cluster == cluster_index):
                break

            # Update the cluster of a point if it meets 1. is a neighbour of the cluster, 2. is unassigned
            # AND 3. is a core point. Then iterate and recompute using the new cluster size
            points_unassigned = np.logical_and.reduce((any_neighbour, np.isnan(cluster), core_points))
            cluster[points_unassigned] = cluster_index

        # Count the number of core points
        n_core = np.sum(cluster == cluster_index)
        
        # Assign the reachable non-core points to the cluster also
        reachable_points_unassigned = np.logical_and.reduce((~core_points, np.isnan(cluster), any_neighbour))
        cluster[reachable_points_unassigned] = cluster_index
        
        # Count the number of non-core points
        n_reachable = np.sum(cluster == cluster_index) - n_core
        if verbose:
            print(f"Cluster {cluster_index:1d}: {n_core:3d} core and {n_reachable:3d} points (total {(n_core + n_reachable):3d}) in {it:2d} iterations")
        
    # Assign outliers to "cluster" 0
    cluster[np.isnan(cluster)] = 0
    
    return cluster


# %% [markdown]
# ## Model usage

# %%
model = DBSCAN(data.values)

# %%
fig, ax = plt.subplots()

sns.scatterplot(x=data["PC1"], y=data["PC2"], hue=model)


plt.show()

# %% [markdown]
# ## Interactive

# %%
features = ["PC1", "PC2"]
cluster_matrix = DBSCAN(data[features].values, .05, 5, verbose=True)

# %%
fig, ax = plt.subplots()

data_at_cluster = lambda f, c: data[f].values[cluster_matrix == c]
for v in np.unique(cluster_matrix):
    if v == 0:
        label = "Outlier"
    else:
        label = f"Cluster {int(v)}"
    ax.scatter(data_at_cluster("PC1", v), data_at_cluster("PC2", v), label=label, s=5)

plt.legend(loc=1)
plt.show()


# %%
def generate_model(data, radius, core_point_threshold, ax, fig):
    # Use PC1 and PC2 to plot
    features = ["PC1", "PC2"]
    
    # Get the clustering after applying the algorithm to the data
    cluster_matrix = DBSCAN(data[features].values, radius, core_point_threshold, verbose=False)
    
    # Macro to extract only the datapoints for a given feature which belong to some cluster
    data_at_cluster = lambda f, c: data[f].values[cluster_matrix == c]
    
    # Clear the axes 
    plt.cla()
    
    # Iterate over each cluster and plot each group with a new artist
    for v in np.unique(cluster_matrix):
        if v == 0:
            label = "Outlier"
        else:
            label = f"Cluster {int(v)}"
        ax.scatter(data_at_cluster("PC1", v), data_at_cluster("PC2", v), label=label)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc=1)
            
    # Trigger a canvas redraw
    fig.canvas.draw()


# %%
# Create a figure for the plot
fig, ax = plt.subplots()

# Call generate_model with the figure and axis 
# so it can draw data onto the canvas
widgets.interact(
    generate_model, 
    radius=widgets.BoundedFloatText(
        min=0, 
        max=1,
        step=0.01,
        description="Radius:",
        value=.25
    ),
    core_point_threshold=widgets.BoundedIntText(
        min=1,
        max=20,
        description="Core point threshold:",
        value=20
    ),
    data=widgets.fixed(data),
    ax=widgets.fixed(ax),
    fig=widgets.fixed(fig)
); None

# %%
