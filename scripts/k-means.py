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


# %% [markdown]
# ## Custom implementation of K-Means

# %%
def KMeansModel(
    data, 
    K, 
    distance = lambda x, p : np.sqrt(np.sum((x - p) ** 2))
):
    """Implementation of K-Means for a dataframe
    Step 1 - Assign all the points to the closest cluster centroid
    Step 2 - Recompute centroids of newly formed clusters
    Step 3 - Repeat until convergence"""
    
    # Define which columns are features
    features = list(data.columns)[:-1]
    
    # Get three random centers
    centers = data[features].sample(n=K).values
    
    # Set a new column to store the cluster that each point belongs to
    data["clusterIndex"] = 0
    
    complete = False
    while not complete:
        # Assign each point to the center
        for point_index, point in data[features].iterrows():
            point_center_dists = {}
            for center_index, center in enumerate(centers):
                point_center_dists[center_index] = distance(point.values, center)
            data.loc[point_index, "clusterIndex"] = min(point_center_dists, key=lambda x: point_center_dists[x])

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
    
    return inertia, centers


# %%
inertia, _ = KMeansModel(copy(data), 3)
inertia

# %% [markdown]
# ## Elbow diagram generation

# %%
inertias = []
centers = []

n_epochs = 5
cluster_range = range(1, 6)

for i in cluster_range:
    print(f"Clustering with k = {i}")
    k_inertia = []
    for ep in tqdm(range(n_epochs)):
        ep_inertia, _ = KMeansModel(copy(data), i)
        k_inertia.append(ep_inertia)
    
    inertias.append(np.mean(k_inertia))

# %%
plt.plot(cluster_range, inertias, marker="o")

plt.xlabel("Number of centroids")
plt.ylabel("Inertia")
plt.xticks(cluster_range)

plt.show()

# %% [markdown]
# A clear elbow point is visible at $k=2$

# %% [markdown]
# # All below this is usless

# %%
pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(feature_data)
print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))

# >> Explained variation per principal component: [0.36198848 0.1920749 ]

print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))

# >> Cumulative variance explained by 2 principal components: 55.41%

# %%
# candidate values for our number of cluster
parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
# instantiating ParameterGrid, pass number of clusters as input
parameter_grid = ParameterGrid({'n_clusters': parameters})
best_score = -1
kmeans_model = KMeans()     # instantiating KMeans model
silhouette_scores = []
# evaluation based on silhouette_score
for p in parameter_grid:
    kmeans_model.set_params(**p)    # set current hyper parameter
    kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p
    ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
    silhouette_scores += [ss]       # store all the scores
    print('Parameter:', p, 'Score', ss)
    # check p which has the best score
    if ss > best_score:
        best_score = ss
        best_grid = p
# plotting silhouette score
plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
plt.xticks(range(len(silhouette_scores)), list(parameters))
plt.title('Silhouette Score', fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

# %%
X = feature_data
y = data.iloc[:,10]


# %%
#reference url "https://www.geeksforgeeks.org/k-means-clustering-introduction/"

def FindColMinMax(X):
    n = len(items[0]);
    minima = [sys.maxint for i in range(n)];
    maxima = [-sys.maxint -1 for i in range(n)];
    
    for item in items:
        for f in range(len(item)):
            if (item[f] < minima[f]):
                minima[f] = item[f];

            if (item[f] > maxima[f]):
                maxima[f] = item[f];
                
    return minima,maxima 



# %%
def InitializeMeans(X, k, cMin, cMax):

    # Initialize means to random numbers between
    # the min and max of each column/feature
    f = len(X[0]); # number of features
    means = [[0 for i in range(f)] for j in range(k)];
    
    for mean in means:
        for i in range(len(mean)):

            # Set value to a random float
            # (adding +-1 to avoid a wide placement of a mean)
            mean[i] = uniform(cMin[i]+1, cMax[i]-1);

    return means;



# %%
def EuclideanDistance(x, y):
    S = 0; # The sum of the squared differences of the elements
    for i in range(len(x)):
        S += math.pow(x[i]-y[i], 2)

    #The square root of the sum
    return math.sqrt(S)



# %%
def UpdateMean(n,mean,X):
    for i in range(len(mean)):
        m = mean[i];
        m = (m*(n-1)+item[i])/float(n);
        mean[i] = round(m, 3);

    return mean;



# %%
def Classify(means,X):

    # Classify item to the mean with minimum distance
    minimum = sys.maxint;
    index = -1;

    for i in range(len(means)):

        # Find distance from item to mean
        dis = EuclideanDistance(X, means[i]);

        if (dis < minimum):
            minimum = dis;
            index = i;
    
    return index;



# %%
def CalculateMeans(k,items,maxIterations=100000):

    # Find the minima and maxima for columns
    cMin, cMax = FindColMinMax(X);
    print(1)
    # Initialize means at random points
    means = InitializeMeans(X,k,cMin,cMax);

    # Initialize clusters, the array to hold
    # the number of items in a class
    clusterSizes= [0 for i in range(len(means))];

    # An array to hold the cluster an item is in
    belongsTo = [0 for i in range(len(X))];

    # Calculate means
    for e in range(maxIterations):

# If no change of cluster occurs, halt
        noChange = True;
        for i in range(len(items)):

            item = items[i];

            # Classify item into a cluster and update the
            # corresponding means.	
            index = Classify(means,item);

            clusterSizes[index] += 1;
            cSize = clusterSizes[index];
            means[index] = UpdateMean(cSize,means[index],item);

            # Item changed cluster
            if(index != belongsTo[i]):
                noChange = False;

            belongsTo[i] = index;

        # Nothing changed, return
        if (noChange):
            break;
    print(means)
    return means;



# %%
def FindClusters(means,items):
    clusters = [[] for i in range(len(means))]; # Init clusters
    
    for item in items:

        # Classify item into a cluster
        index = Classify(means,item);

        # Add item to cluster
        clusters[index].append(item);

    return clusters;



# %% [markdown]
# We got elbow point as 3 means we need to have 3 clusters but as we have only 2 classs.

# %%
#reference url "https://www.machinelearningplus.com/predictive-modeling/k-means-clustering/" 
import seaborn as sns
X
n_iter=50
m=X.shape[0]
n=X.shape[1] 

# %%
#computing the initial centroids randomly
K=3
import random

# creating an empty centroid array
centroids=np.array([]).reshape(n,0) 

# creating 5 random centroids
for k in range(K):
    centroids=np.c_[centroids,X[random.randint(0,m-1)]]
    output={}

# creating an empty array
euclid=np.array([]).reshape(m,0)

# finding distance between for each centroid
for k in range(K):
       dist=np.sum((X-centroids[:,k])**2,axis=1)
       euclid=np.c_[euclid,dist]
# storing the minimum value we have computed
minimum=np.argmin(euclid,axis=1)+1




# %%
# computing the mean of separated clusters
cent={}
for k in range(K):
    cent[k+1]=np.array([]).reshape(2,0)

# assigning of clusters to points
for k in range(m):
    cent[minimum[k]]=np.c_[cent[minimum[k]],X[k]]
for k in range(K):
    cent[k+1]=cent[k+1].T

# computing mean and updating it
for k in range(K):
     centroids[:,k]=np.mean(cent[k+1],axis=0)

# %%
for i in range(n_iter):
      euclid=np.array([]).reshape(m,0)
      for k in range(K):
          dist=np.sum((X-centroids[:,k])**2,axis=1)
          euclid=np.c_[euclid,dist]
      C=np.argmin(euclid,axis=1)+1
      cent={}
      for k in range(K):
           cent[k+1]=np.array([]).reshape(2,0)
      for k in range(m):
           cent[C[k]]=np.c_[cent[C[k]],X[k]]
      for k in range(K):
           cent[k+1]=cent[k+1].T
      for k in range(K):
           centroids[:,k]=np.mean(cent[k+1],axis=0)
      final=cent


# %%
def init_centroids(k, X):
    arr = []
    for i in range(k):
        cx1 = np.random.uniform(X)
        cx2 = np.random.uniform(X)
        arr.append([cx1, cx2])
    return np.asarray(arr)


# %%
def dist(a, b):
    return np.sqrt(sum(np.square(a-b)))


# %%
def assign_cluster(k, X, cg):
    cluster = [-1]*len(X)
    for i in range(len(X)):
        dist_arr = []
        for j in range(k):
            dist_arr.append(dist(X[i], cg[j]))
        idx = np.argmin(dist_arr)
        cluster[i] = idx
    return np.asarray(cluster)


# %%
def compute_centroids(k, X, cluster):
    cg_arr = []
    for i in range(k):
        arr = []
        for j in range(len(X)):
            if cluster[j]==i:
                arr.append(X[j])
        cg_arr.append(np.mean(arr, axis=0))
    return np.asarray(cg_arr)


# %%
def measure_change(cg_prev, cg_new):
    res = 0
    for a,b in zip(cg_prev,cg_new):
        res+=dist(a,b)
    return res


# %%
def show_clusters(X, cluster, cg):
    pass


# %%
def k_means(k, X):
    cg_prev = init_centroids(k, X)
    cluster = [0]*len(X)
    cg_change = 100
    while cg_change>.001:
        cluster = assign_cluster(k, X, cg_prev)
        show_clusters(X, cluster, cg_prev)
        cg_new = compute_centroids(k, X, cluster)
        cg_change = measure_change(cg_new, cg_prev)
        cg_prev = cg_new
    return cluster

cluster = k_means(3, X)
