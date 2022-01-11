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

# %% [markdown]
# ### todo:
# - Make the plot interactive to view by point probability
# - 3d contour plot of the gaussians

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Gaussian Mixture Model
#
# This method assumes that in each class the data are distributed normally across each of the features, in other words the distribution sampled by the data is expressable as a composition of Gaussians.
#
# The composite Gaussian mixture population distribution can be expressed:
# $$ P(x) = \sum_{k=1}^K \Pi_k N \left( x_i | \mu_k, \Sigma^k \right) $$
# To normalize this we must normalize the weights $\Pi_k$, and each normal distrinution such that the integral over all space is unitary.
#
# The parameters of this distribution can be written $\theta = \left\{ \pi_n, \mu_n, \Sigma_n \right\}$, the weight of the gaussian and the mean and standard deviation of the gaussian. The aim is to find parameters $\theta$ which maximize the likelihood of the observed data. This is the only gaussian mixture of interest.
#
# WE can learn these parameters using the Expectation Maximization algorithm. The size of a cluster is given by the eigenvalues of the covariance matrix corresponding to the cluster. The model captures the location and size of each cluster. Therefore the probability of observing soem point $x$ is given:
# $$ P\left(x\right) = \sum_{k=1}^K\left(\Pi_k N\left(\mu_k,\Sigma_k\right)\right) $$
# In other words, it is the weighted sum of the probability at each cluster.
#
# The algorithm for EM follows:
# 1. Fix the number of clusters $K$
# 2. Initialize the parameters $\theta = \left\{ \Pi_k, \mu_k, \Sigma_k \right\},\: k=1,\ldots,K$
# 3. Calculate the responsibility for each cluster. There is a responsibility $\gamma_{kj}$ of cluster $k$ and datapoint $x_j$ which dictates how much of the overall probability of observing $x_j$ comes from $k$.
# 4. Calculate the weights $\Pi_k$ from the responsibilities. The overall weight $\Pi_k$ is the likelihood of $k$ to be relevant to some point $x$.
# 5. Calculate the new mean and covariance matrix from the the weights and responsibilities.
# 6. Repeat until convergence.
#
# This could also be considered like the problem on the EM wikipedia page, with $x$ being a two-featured vector and $z ~ \text{Categorical}(k, \phi)$. In our case, $k=2$ and $\phi_k$, the population probability of benign vs malignant, is unknown and correpsoinding to the weight of each cluster. Interestingly $z$ behaves like a latent variable to this data, although it can be directly observed it is known at the time of data collection. 
#
# Our data are labelled, meaning it should be possible to use maximum-likelihood estimation to directly calculate the best parameters. However if not, then EM can also estimate $z$.
#
# Note that modelling the underlying data we have as Gaussian is not particuarly valid; in a Normal distribution there is no restriction to the values which may possibly be sampled; whereas in each of our features we define the minimum value to be 1 and the maximum to be 10. 

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

# %% [markdown] tags=[]
# # Data handling

# %% tags=[]
data = pd.read_csv("../data/data-pca.csv")
d2_cols = ["PC1", "PC2", "class"]
data = full_data[d2_cols]


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
def compute_likelihood(x, model):
    point_likelihoods = []
    for i, cluster in enumerate(model.clusters):
        probability = evaluate_pdf(x, Gaussian(cluster.mean, cluster.cov))
        point_likelihoods.append(model.cluster_weights[i] * probability)
    return np.log(np.sum(point_likelihoods))


# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## EM algorithm

# %%
def GMM(data, K, seed=42, maxiter=1e3, stop_threshold=1e-3):  
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
    
    converged = False
    complete = False
    while not complete:     
        """
        E-step: Calculate the posterior probabilities given the clusters we have.
                Compute the likelihood given these clusters and weights
        """
        # Calculate the responsibility matrix  
        new_resp = np.stack(
            data[features].apply(
                responsibility, 
                axis=1, 
                model=model
            ).values
        )
        
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
            converged = True
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
        print(f"Iteration {model.it}", end="\r")

    
    print(f"Completed in {model.it} iterations")
    return model

# %% [markdown]
# # Fit and result

# %%
gmm_model = GMM(data[d2_cols], 2)
gmm_model.plot_likelihood()

# %% [markdown]
# ## Visualise clusters

# %%
from functools import partial

# %%
# Generate partial functions to be used with vectorizing
samplers = [partial(evaluate_pdf, gauss=Gaussian(i.mean, i.cov)) for i in gmm_model.clusters]
class_data = lambda f, l: data[data["class"] == l][f]


# %% tags=[]
# From http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html
def plot_gaussians(model, resolution=1e-1):
    x, y = np.mgrid[-30:0:.1, -10:15:.1]
    position = np.empty(x.shape + (2,))
    position[:, :, 0] = x
    position[:, :, 1] = y

    plt.figure(figsize = (15, 6))
    plt.scatter(class_data("PC1", 0), class_data("PC2", 0), label="Benign cases")
    plt.scatter(class_data("PC1", 1), class_data("PC2", 1), label="Malignant cases")

    for i in range(2):
        z = np.apply_along_axis(samplers[i], 2, position)
        plt.contour(x, y, z, colors="black")

    plt.xlim([-25, 0])
    plt.ylim([-10, 15])

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc=1)
    plt.show()


# %% tags=[]
plot_gaussians(gmm_model)
