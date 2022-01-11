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
from statistics import NormalDist

import matplotlib.pyplot as plt

# %% tags=[]
full_data = pd.read_csv("../data/data-pca.csv")
cols = ["PC1", "PC2", "class"]
data = full_data[cols]


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
        plt.plot(range(self.it-1), self.trace_likelihood, label="Likelihood")
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
from scipy.stats import multivariate_normal


# %%
def responsibility(x, model):
    # Function to calculate the responsibility vector for a given x
    # This can and should be vectorized further
    resp = np.zeros((model.K))
    for i, cluster in enumerate(model.clusters):
        # resp[i] = model.cluster_weights[i] * evaluate_pdf(x.values, cluster)
        resp[i] = model.cluster_weights[i] * multivariate_normal(cluster.mean, cluster.cov).pdf(x.values)

    return resp/resp.sum()


# %%
def compute_likelihood(x, model):
    point_likelihoods = []
    for i, cluster in enumerate(model.clusters):
        probability = multivariate_normal(cluster.mean, cluster.cov).pdf(x)
        point_likelihoods.append(model.cluster_weights[i] * probability)
    return np.log(np.sum(point_likelihoods))


# %%
def GMM(data, K, prior=None, seed=40, maxiter=1e2):    
    model = Model(K=K)
    
    # Define which columns are features
    features = list(data.columns)[:-1]
    n_dims = len(features)  
    
    # Get K random centers (not duplicates)
    center_means = data[features].sample(n=K)
    while center_means.duplicated().any().any():
        center_means = data[features].sample(n=K)    
    
    np.random.seed(seed)
    chosen = np.random.choice(data.shape[0], model.K, replace = False)
    center_means = data[features].values[chosen]
    
    # Initialise the centers with mean=m and sd=1
    model.clusters = [
        Gaussian(
            i, 
            np.cov(data[features].values, rowvar=False)
        ) 
        for i in center_means
    ]
            
    # Initialise the cluster weights and normalize
    model.cluster_weights = np.ones((model.K)) / model.K    
    
    # Matrix to store the responsibilities
    model.responsibility = np.zeros((data.shape[0], K))
    
    # Initialize the likelihood
    model.likelihood = -np.inf
    
    if prior:
        model = prior
        
    complete = False
    while not complete:     
        ##########
        # E STEP #
        ##########
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
        
        ##########
        # M STEP #
        ##########
        new_gaussians = []
        new_weights = []
        for i in range(model.K):
            x = data[features].values
            r = new_resp[:, i]
            sum_r = np.sum(r)
            N_k = len(r)
            
            # Calculate weights
            pi = sum_r / N_k
            
            # Calculate centers
            rx = np.multiply(np.vstack([r] * n_dims).T, x)
            mu = (1 / sum_r) * np.sum(rx, axis=0) 
                                    
            # Calculate covariance matrices
            outer_product = lambda x: np.outer(x, x.T)
            x_outer_products = np.apply_along_axis(outer_product, 1, x)
            r_broadcast = r.reshape(r.shape[0], 1, 1)
            sigma = (1 / sum_r) * np.sum((r_broadcast * x_outer_products), axis=0)
                        
            # Save the new weight
            new_weights.append(pi)
            
            # Create new gaussians
            new_gaussians.append(Gaussian(mean=mu, cov_mat=sigma))
        
        # Update the model
        model.clusters = new_gaussians
        model.cluster_weights = new_weights
        model.responsibility = new_resp
        
        # Stop condition
        if np.abs(likelihood - model.likelihood) < 1e-2 or model.it >= maxiter:
            complete = True
            
        model.likelihood = likelihood
        
        model.trace_mean.append([i.mean for i in model.clusters])
        model.trace_likelihood.append(likelihood)
        print(f"Iteration {model.it}", end="\r")
        
        # Update the iteration counter
        model.it += 1

    
    print(f"Completed in {model.it-1} iterations")
    return model

# %%
model2 = GMM(copy(data), 2, maxiter=40)

# %% tags=[]
model2.plot_likelihood()

# %% tags=[]
import ipywidgets as widgets

# %% tags=[]
# From http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html
from scipy.stats import multivariate_normal
def plot_gaussians(x_min, x_max, y_min, y_max, model):
    x, y = np.mgrid[x_min:x_max:.1, y_min:y_max:.1]
    position = np.empty(x.shape + (2,))
    position[:, :, 0] = x
    position[:, :, 1] = y

    plt.figure(figsize = (15, 6))
    plt.scatter(data["PC1"], data["PC2"])
    for i in range(2):
        # plt.subplot(1, 3, i + 1)
        z = multivariate_normal(model.clusters[i].mean, model.clusters[i].cov).pdf(position)
        plt.contour(x, y, z)

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# %% tags=[]
widgets.interact(
    plot_gaussians,
    x_min=widgets.IntSlider(min=-100, max=100, value=-40),
    x_max=widgets.IntSlider(min=-100, max=100, value=10),
    y_min=widgets.IntSlider(min=-100, max=100, value=-10),
    y_max=widgets.IntSlider(min=-100, max=100, value=10),
    model=widgets.fixed(model2)
)


# %% [markdown]
# ## Visualise clusters

# %% tags=[] jupyter={"source_hidden": true}
def plot_contours(data, means, covs, title):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-25.0, 0.0, delta)
    y = np.arange(-10.0, 10.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors = col[i])

    plt.title(title)
    plt.tight_layout()


# %%
features = ["PC1", "PC2"]
