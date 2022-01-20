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
# # This text is mostly bullshit, needs rewriting

# %%
# %matplotlib widget

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### todo:
# - [x] Make the plot interactive to view by point probability
# - [x] 3d contour plot of the gaussians
# - Try GMM on the data that have not been preprocessed (will probably fail badly from singluar matrices)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Gaussian Mixture Model

# %% [markdown]
# ## Expectation-Maximization
#
# This follows the algorighm
#
# - Define $K$
# - Initialise $\theta_k$ and the probabilities $\pi_k$
# - For each $x_i$ calculate the responsibilities of each gaussian (normalized)
# - Using these compute the means and covariance matrices
# - Calculate the weights using the responsibilities

# %%
import pandas as pd
import numpy as np

from copy import copy
import matplotlib.pyplot as plt
import ipywidgets as widgets

from functools import partial

# %% [markdown] tags=[]
# # Data handling

# %% tags=[]
data = pd.read_csv("../data/data-pca.csv")
# clustering_cols = ["PC1", "PC2", "class"]
clustering_data = data


# %% [markdown] tags=[]
# # GMM definition

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Object definitions

# %%
class Model():
    def __init__(self, K=0, data=None, clusters=None, cluster_weights=None):
        self.K = K
        self.data = data
        self.clusters = clusters
        self.cluster_weights = cluster_weights
        self.responsibility = None
        self.trace_likelihood = []
        self.trace_mean = []
        self.it = 1
    
    def plot_likelihood(self):
        plt.plot(range(self.it), self.trace_likelihood, label="Likelihood")
        plt.legend()


# %%
class Gaussian():
    # Should provide a n x 1 dim vector for mean
    # n x n vector for cov_mat
    def __init__(self, mean, cov_mat):
        self.mean = mean
        self.cov = cov_mat
    
    def __eq__(self, o):
        # This raises some numpy error somehow?
        # assert getattr(o, "mean", False) and getattr(o, "cov", False)
        return (self.mean == o.mean).all() and (self.cov == o.cov).all()
    
    def __repr__(self):
        return f"<Gaussian with mean {self.mean}>"


# %% [markdown] tags=[]
# ## Statistical functions

# %%
def evaluate_pdf(x, gauss):
    # Assert that the covariance matrix is invertible (positive definite?)
    # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
    d = x - gauss.mean
    inv = np.linalg.inv(gauss.cov)
    num = np.exp(-0.5 * (d.T @ inv @ d))
    den = np.sqrt(((2 * np.pi) ** np.linalg.matrix_rank(gauss.cov)) * np.linalg.det(gauss.cov))
    return num / den


# %%
def responsibility(x, model):
    # Function to calculate the responsibility vector for a given x
    # This can and should be vectorized further
    resp = np.zeros((model.K))
    for i, cluster in enumerate(model.clusters):
        resp[i] = model.cluster_weights[i] * evaluate_pdf(x.values, cluster)

    return resp/resp.sum()


# %%
def point_likelihood(x, model):
    likelihoods = np.empty(len(model.clusters))
    for i, cluster in enumerate(model.clusters):
        likelihoods[i] = model.cluster_weights[i] * evaluate_pdf(x, Gaussian(cluster.mean, cluster.cov))
    return likelihoods


# %%
def compute_likelihood(x, model):
    return np.log(np.sum(point_likelihood(x, model)))


# %%
def compute_likelihood_old(x, model):
    point_likelihoods = []
    for i, cluster in enumerate(model.clusters):
        probability = evaluate_pdf(x, Gaussian(cluster.mean, cluster.cov))
        point_likelihoods.append(model.cluster_weights[i] * probability)
    return np.log(np.sum(point_likelihoods))


# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## EM algorithm

# %%
def GMM(data, K, seed=42, maxiter=1e3, stop_threshold=1e-3, verbose=True):  
    # Set the model seed
    np.random.seed(seed)
    
    # Create a model to store the data in
    model = Model(K=K)
    
    # Define which columns are features and data properties
    features = list(data.columns)[:-1]
    X = data[features].values    
    n_instances, n_dims = X.shape
    
    # Get K random centers 
    indices = np.random.choice(n_instances, K, replace=False)
    center_means = X[indices]
    
    # Initialise the centers using the covariance of the entire dataset
    # Can also initialise in other ways
    model.clusters = [
        Gaussian(
            i, 
            np.cov(X, rowvar=False)
        ) 
        for i in center_means
    ]
            
    # Initialise the cluster weights and normalize
    model.cluster_weights = np.ones(K) / K    
    
    # Matrix to store the responsibilities
    model.responsibility = np.zeros((n_instances, K))
    
    # Initialize the likelihood to ensure the stop condition
    # is not met on the first iteration
    model.likelihood = -np.inf
    
    model.converged = False
    complete = False
    while not complete:     
        """
        E-step: Calculate the posterior probabilities given the clusters we have.
                Compute the likelihood given these clusters and weights
        """
        # Calculate the responsibility matrix  
        try:
            new_resp = np.stack(
                data[features].apply(
                    responsibility, 
                    axis=1, 
                    model=model
                ).values
            )
        except np.linalg.LinAlgError as e:
            if verbose: print(repr(e))
            return model
            
        # Compute the log likelihood 
        likelihood = np.sum(
            data[features].apply(
                compute_likelihood,
                axis=1,
                model=model
        ).values)
        
        """
        M-step: Update the mean, covariance matricies and weights of the Gaussians to
                increase the value of the log-likelihood, based on the posterior 
                probability calcualted.
        """
        new_gaussians = []
        new_weights = []
        for i in range(model.K):
            x = data[features].values
            r = new_resp[:, i]
            sum_r = np.sum(r)
            N_k = x.shape[0]
            
            # Calculate weights
            pi = sum_r / N_k
            
            # Calculate centers
            rx = np.multiply(np.vstack([r] * n_dims).T, x)
            mu = (1 / sum_r) * np.sum(rx, axis=0) 
                                    
            # Calculate covariance matrices
            outer_product = lambda x: np.outer(x, x.T)
            x_outer_products = np.apply_along_axis(outer_product, 1, (x - mu))
            r_broadcast = r.reshape(r.shape[0], 1, 1)
            sigma = (1 / sum_r) * np.sum((r_broadcast * x_outer_products), axis=0)
                        
            # Save the new weight
            new_weights.append(pi)
            
            # Create new gaussians
            new_gaussians.append(Gaussian(mean=mu, cov_mat=sigma))      
        
        
        # Check if the stop conditions of the model have been reached
        if np.abs(likelihood - model.likelihood) < stop_threshold:
            model.converged = True
            complete = True
        elif model.it >= maxiter:
            complete = True
        else:
            model.it += 1 # If not then update the iteration counter
        
        # Save the new values to the model
        model.clusters = new_gaussians
        model.cluster_weights = new_weights
        model.responsibility = new_resp
        model.likelihood = likelihood
        
        # Save the likelihood history for visualisation
        model.trace_likelihood.append(likelihood)
        
        # Display the iteration and update the counter
        if verbose: print(f"Iteration {model.it}", end="\r")

    
    if verbose: print(f"Completed in {model.it} iterations")
    return model

# %% [markdown]
# # Fit and result

# %%
K = 2

gmm_model = GMM(clustering_data, K)
gmm_model.plot_likelihood()


# %% [markdown]
# Converting the Gaussian functions into the feature space

# %% [markdown] tags=[]
# ## Matplotlib functions

# %%
# Generate partial functions to be used with vectorizing

# %% tags=[]
# From http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html
def plot_gaussians(model, data, ax, resolution=.1, features=["PC1", "PC2"], pc3_coordinate=0, fig=None):
    samplers = [partial(evaluate_pdf, gauss=Gaussian(i.mean, i.cov)) for i in model.clusters]
    class_data = lambda f, l: data[data["class"] == l][f]
    
    if not fig:
        fig = plt.gcf()
    plt.cla()
    x, y = np.mgrid[-3:3:resolution, -3:3:resolution]
    position = np.empty(x.shape + (3,))
    position[:, :, 0] = x
    position[:, :, 1] = y
    position[:, :, 2] = pc3_coordinate

    ax.scatter(class_data(features[0], 0), class_data(features[1], 0), label="Benign cases", s=0.4)
    ax.scatter(class_data(features[0], 1), class_data(features[1], 1), label="Malignant cases", s=0.4)

    for i in range(model.K):
        z = np.apply_along_axis(samplers[i], 2, position)
        ax.contour(x, y, z, colors="black")

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Contour plot of the clusters fitted.")
    
    ax.legend(loc=1)
    fig.canvas.draw()


# %%
def drawax3d(model, ax, resolution=.1, pc3_coordinate=0, fig=None):
    samplers = [partial(evaluate_pdf, gauss=i) for i in model.clusters]
    # Create a plot and 3d axes
    range_pc1 = range(-3, 3)
    range_pc2 = range(-3, 3)
    
    if not fig:
        fig = plt.gcf()
    plt.cla()
    
    pc1, pc2 = np.mgrid[range_pc1[0]:range_pc1[-1]:resolution, range_pc2[0]:range_pc2[-1]:resolution]
    position = np.empty(pc1.shape + (3,))
    position[:, :, 0] = pc1
    position[:, :, 1] = pc2
    position[:, :, 2] = pc3_coordinate

    for i in range(model.K):
        z = np.apply_along_axis(samplers[i], 2, position)
        ax.plot_surface(pc1, pc2, z, cmap="coolwarm", linewidth=0, antialiased=True, rstride=1, cstride=1)


    # Set the rotation and axes labels
    # ax.view_init(30, rotation)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Probability")
    ax.set_title("Visualisation of the Gaussian mixture surface.")
    fig.canvas.draw()


# %% [markdown]
# ## Visualisations

# %%
fig = plt.figure(figsize=plt.figaspect(.5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

plot_gaussians(gmm_model, clustering_data, ax1, resolution=.075)
drawax3d(gmm_model, ax2, resolution=.1, pc3_coordinate=0)

plt.show()

# %%
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(projection="3d")

widgets.interact(
    drawax3d,
    model=widgets.fixed(gmm_model),
    data=widgets.fixed(clustering_data),
    ax=widgets.fixed(ax),
    resolution=widgets.FloatSlider(min=.1, max=1, step=0.05),
    features=widgets.fixed(["PC1", "PC2"]),
    pc3_coordinate=widgets.FloatSlider(min=-1, max=1, step=0.05),
    fig=widgets.fixed(fig)
)

# %%
from scipy import stats

means = np.stack([i.mean for i in clusters])
covs = np.stack([i.cov for i in clusters])

fig, ax = plt.subplots()

for i in gmm_model.clusters:
    x1 = np.linspace(i.mean[0] - 3 * i.cov[0, 0], i.mean[0] - 3 * i.cov[0, 0], 100)
    y1 = stats.norm.pdf(x1, i.mean[0], i.cov[0, 0])
    
    print(x1)
    
    ax.plot(x1, y1)
    
plt.show()

# %%
from utility import transform, recover

# %%
clusters = gmm_model.clusters

# %%
means = np.stack([i.mean for i in clusters])
covs = np.stack([i.cov for i in clusters])

# %%
means.shape

# %%
means_padded = np.pad(means, ((0, 0), (0, 6))) # pad the values with the column means, 0 
means_recovered = recover(means_padded)
means_recovered[1, :]

# %% [markdown]
# # Specificity sensetivity curve
#
# Generated by artificially favouring one cluster over another

# %%
from functools import partial
from utility import specificty_sensetivity


# %%
def make_prediction(x, model, decision_threshold=0.5):
    # sample probability point belongs to cluster 1
    # sample probability point belongs to cluster 2
    likelihoods = point_likelihood(x, model)
    return (decision_threshold * likelihoods[1]) - ((1 - decision_threshold) * likelihoods[0]) > 0
    # return likelihoods[1] > likelihoods[0] # 1 for cluster 2, 0 for cluster 1 


# %%
features = ["PC1", "PC2", "PC3"]

pr = partial(make_prediction, model=gmm_model)

predictions = np.apply_along_axis(pr, 1, data[features]).astype(int)
labels = data["class"].values
diff = np.stack((labels, predictions), 1)

specificty_sensetivity(diff)

# %%
