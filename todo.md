# Project todos.

- [ ] Conduct a literature search of all the papers written on the WBC dataset and on discrere intervaled data in
	general. Upload all papers to the `Report-1` repository
- [ ] Decide on what type of normalization and dimensionallity reduction to use.
	- Is it valid to treat our data as real-valued instead of being discrete intervals? 
	- Does dimensionallity reduction such as PCA or LDA work on the type of data that we have? 
	- Can our data be transformed to be normally distributed, or does it follow another distribution? 
	- How do we treat the fact that our data come from two classes; can you split the dataset into one set for benign and another for malignant?
	- How does that change the analyses we can do?
- [x] Generate visualisations of the features 
- [ ] Implement some kind of dimensionallity reduction
- [ ] Generate visualisations of the reduced dimension dataset
- [x] Implement K-Means
- [ ] Implement another clustering method (GMM, DBSCAN, heriachical clustering)
- [ ] Generate visualisations of clustering techniques
- [ ] Look into autoencoders or other ANN-based unsupervised techniques

## Dimensionallity reuction

This might be difficult to achieve on our dataset since the data we have is in a discrete range from 1 - 10. This is
discrete interval data - there is no natural "zero" value which is required for ratio data. Other studies have treated
this data as being real-valued, and this is perhaps a good angle to take. However, normalizing the dataset is not
possible as none of the features are normally distributed. This isnt an assumption for PCA, however for PCA to work
properly the features need to be entirely linearly independent. This is probably not the case, and so care must be
taken.

## Clustering

K-Means is already implemented, with the optimal number of clusters being 2 or 3. It is impossible to visualise the
position of these clusters without dimension reduction, however the purity of each is around 0.95 meaning they do
achieve good separation of the two classes. Finding out which features are the most important to a classification of
benign/malignant is an important goal of the research

GMM model is next to be implemented, which will hopefully improve on the result from before. However for this model
having visualisations is even more important than in the K-means case, so PCA, ICA or LDA must be implemented first. 

## Visualisations

The graphics generated will be an important part of the paper submitted. Therefore reducing the number of dimensions to
two is very important. If PCA is used then the percentage contribution of each feature to PC1 and PC2 will assess its
impact on the diagnosis. However there are other ways to measure the importance of a feature on a class and we should
consider those also. See the graphics already present in the preprocessing notebook for some inspiration.

## Other unsupervised techniques

Look into the usage of
- Autoencoders
- Adaptive Resonance Theory (???)
- Probabilistic techniques (PGN?)
