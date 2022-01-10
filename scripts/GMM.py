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

# %% jupyter={"source_hidden": true} tags=[]
full_data = pd.read_csv("../data/data-pca.csv")
cols = ["PC1", "PC2", "class"]
data = full_data[cols]
data


# %%
class Model():
    def __init__(self, K=0, data=None, clusters=None, cluster_weights=None):
        self.K = K
        self.data = data
        self.clusters = clusters
        self.cluster_weights = cluster_weights
        self.responsibility = None


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
def responsibility(x, model):
    # Function to calculate the responsibility vector for a given x
    # This can and should be vectorized further
    resp = np.zeros((model.K))
    for i, cluster in enumerate(model.clusters):
        resp[i] = model.cluster_weights[i] * evaluate_pdf(x.values, cluster)
    return resp/resp.sum()


# %%
evaluate_pdf(np.array([1, 1]), Gaussian(mean=np.array([6, 6]), cov_mat=np.identity(2)))


# %%
def r(x, ga, wa, gb, wb):
    return wa * evaluate_pdf(x, ga) / (wa * evaluate_pdf(x, ga) + wb * evaluate_pdf(x, gb))

x = np.array([0, 0])
ga = Gaussian(np.array([1, 1]), 5 * np.identity(2))
gb = Gaussian(np.array([5, 5]), 5 * np.identity(2))
wa = wb = 0.5


r(x, gb, wb, ga, wa)

# %%
x = np.array([0, 0])
u = 0.60281916

np.outer((x - u), (x - u).T)


# %%
def GMM(data, K, prior=None):
    model = Model(K=K)
    
    # Define which columns are features
    features = list(data.columns)[:-1]
    
    # Get K random centers (not duplicates)
    center_means = data[features].sample(n=K)
    while center_means.duplicated().any().any():
        center_means = data[features].sample(n=K)    
    
    # Initialise the centers with mean=m and sd=1
    model.clusters = [Gaussian(i, 5 * np.identity(len(features))) for i in center_means.values]
            
    # Initialise the cluster weights and normalize
    model.cluster_weights = np.random.uniform(size=(K, 1))
    model.cluster_weights = model.cluster_weights/model.cluster_weights.sum()
    
    # Matrix to store the responsibilities
    model.responsibility = np.zeros((data.shape[0], K))
    
    if prior:
        model = prior
    
    complete = False
    it = 0
    while not complete:
        # complete = True
        # print(f"Iteration {it} ------------------")
        # print("Model clusters (means)")
        # print([i.mean for i in model.clusters])
        # print("Model clusters (covariances)") 
        # for i in model.clusters:
        #     print("------------")
        #     print(i.cov)
        # print("------------")
        # print("Responsibility")
        # print(model.responsibility)
        # print("Weights")
        # print(model.cluster_weights)
        # print("FullIter")
        
        # Calculate the responsibility matrix  
        # Passes each row of data[features] as the first argument
        model.responsibility = np.stack(
            data[features].apply(
                responsibility, 
                axis=1, 
                model=model
            ).values
        )
            
        # The responsiblilty matrix can be used to compute the means
        new_means = np.zeros((model.K, len(features)))
        # resp_data = np.hstack((model.responsibility, data[features].values))
        for i, cluster in enumerate(model.clusters):
            sum_scaled_data = np.zeros((len(features)))
            sum_responsibilities = 0
            for j, x in enumerate(data[features].values):
                sum_scaled_data += x * model.responsibility[j, i]
                sum_responsibilities += model.responsibility[j, i]
            # print(f"Cluster {i}")
            # print(sum_scaled_data)
            # print(sum_responsibilities)
            new_means[i, :] = sum_scaled_data / sum_responsibilities
        
        # Calculate the covariance matrices
        new_covariance_matrices = np.zeros((len(features), len(features), model.K))
        for i, cluster in enumerate(model.clusters):
            sum_covariance_matrix = np.zeros((len(features), len(features)))
            sum_responsibilities = 0
            for j, x in enumerate(data[features].values):
                mu = new_means[i]
                d = x - mu
                resp = model.responsibility[j, i]
                sum_covariance_matrix += resp * (np.outer(d, d.T))
                sum_responsibilities += resp
            new_covariance_matrices[:, :, i] = sum_covariance_matrix / sum_responsibilities
            
        # Calculate the new weights
        # new_weights = np.zeros((model.K, 1))
        new_weights = np.sum(model.responsibility, axis=0) / model.responsibility.shape[0]
                
        # Create new clusters
        new_clusters = [
            Gaussian(
                new_means[i, :], 
                new_covariance_matrices[:, :, i]
            )
            for i in range(model.K)
        ]
            
        # Check if the mean has changed  
        if new_clusters == model.clusters:
            complete = True
        
        # Update the centers
        for i in range(len(model.clusters)):
            model.clusters[i] = Gaussian(
                mean=new_means[i, :], 
                cov_mat=new_covariance_matrices[:, :, i]
            ) 
        
        # Update the weights
        model.cluster_weights = new_weights
            
        it += 1
    print(f"Completed in {it} iterations")
    return model


# %%
model = GMM(copy(data).sample(n=680), 2)

# %%
model.clusters

# %%
from scipy.stats import multivariate_normal

# %%
x, y = np.mgrid[-30:10:.1, -30:10:.1]
position = np.empty(x.shape + (2,))
position[:, :, 0] = x
position[:, :, 1] = y

# different values for the covariance matrix
covariances = [ [[1, 0], [0, 1]], [[1, 0], [0, 3]], [[1, -1], [-1, 3]] ]
titles = ['spherical', 'diag', 'full']

# %%
position.shape

# %%
plt.figure(figsize = (15, 6))
plt.scatter(data["PC1"], data["PC2"])
for i in range(2):
    # plt.subplot(1, 3, i + 1)
    z = multivariate_normal(model.clusters[i].mean, model.clusters[i].cov).pdf(position)
    plt.contour(x, y, z)
    plt.xlim([-30, 0])
    plt.ylim([-10, 12])

plt.show()

# %%

# %% [markdown]
# ## Gaussian modelling

# %% [markdown]
# ## Visualise clusters

# %%
