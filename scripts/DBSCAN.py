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

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from functools import partial

# %%
data = pd.read_csv("../data/data-pca.csv")

# %%
plt.scatter(data["PC1"], data["PC2"], c=data["class"])

# %%
data["PC1"].std()

# %%
radius = 2
minpts = 10


# %%
def distance(a, b):
    return np.sqrt(np.sum((a-b) ** 2))


# %%
def find_core_points(x, data, radius, min_pts):
    d = partial(distance, b=x)
    distances = np.apply_along_axis(d, 1, data)
    return min_pts < np.sum(distances < radius)


# %%
def compute_distance_to_all_points(x, data):
    d = partial(distance, b=x)
    distances = np.apply_along_axis(d, 1, data)
    # print(distances.reshape(1, data.shape[0]).shape)
    return distances


# %%
# Find all the core points

# %%
d = data[["PC1", "PC2"]].values
cl = partial(find_core_points, data = d, radius=radius, min_pts=minpts)
core_points = np.apply_along_axis(cl, 1, d)

# %%
d[core_points].shape

# %%
n_instances, n_dims = d.shape

# %%
np.random.seed(42)

# Define an empty set of cluster indices
cluster = np.full((n_instances), np.nan)

# Define masks to access the assigned and unassigned core points
assigned_core_points = np.logical_and(core_points, ~np.isnan(cluster))
unassigned_core_points = np.logical_and(core_points, np.isnan(cluster))

# Compute the distance matrix mapping each point to eachother
compute_distance = partial(compute_distance_to_all_points, data=d)
distance_matrix = np.apply_along_axis(compute_distance, 1, d)

# %%
cluster_index = 0
while d[unassigned_core_points].shape[0] > 0:
    # Set the random seed
    np.random.seed(42)

    # Define the index of this cluster
    cluster_index += 1
    
    # Check there are unassigned core points to sample
    sample_pool = np.where(np.logical_and(core_points, np.isnan(cluster)))[0]
    if len(sample_pool) == 0:
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

    print(f"Cluster {cluster_index} core points added: {np.sum(cluster == cluster_index)} in {it} iterations")
    
    # Assign the reachable non-core points to the cluster also
    reachable_points_unassigned = np.logical_and.reduce(
        (~core_points, np.isnan(cluster), any_neighbour)
    )
    print(np.sum(reachable_points_unassigned))
    cluster[reachable_points_unassigned] = cluster_index
    
    print(f"Assigned {np.sum(cluster == cluster_index)} points total")

# Assign outliers 0
cluster[np.isnan(cluster)] = 0

# %%
cluster

# %%
plt.scatter(data["PC1"], data["PC2"], c=cluster)

# %%
np.sum(cluster == 5)

# %%
# Form a cluster from a core point
# Define all the points in its neighbourhood
# Add neighbouring core points to the cluster
# Repeat until there are no more core points
# Define all non-core points which are within the radius of the core points assigned to the cluster
# Add these to the cluster

# Repeat until every core point has been assigned to a cluster
# Mark the remaining points as outliers

# %%
# Compute a distance matrix for each point
# Create a core points mask
# while there are unassigned core points:
    # Pick a random core point
    # Assign it to a cluster
        # Find all the core points adjacent to the cluster
        # Add those to the cluster
        # Repeat until no more core points
    # Find all the non-core points adjacent to the cluster
    # Add them too
