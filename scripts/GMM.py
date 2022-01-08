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

# %% [markdown]
# ## Gaussian modelling

# %% [markdown]
# ## Visualise clusters
