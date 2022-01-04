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
# # Data preprocessing

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import re

# %%
# %matplotlib inline

# %% [markdown]
# Data will be processed and output as a new csv.

# %%
feature_names = [
    "clumpThickness",
    "uniformityOfCellSize",
    "uniformityOfCellShape",
    "marginalAdhesion",
    "singleEpithelialCellSize",
    "bareNuclei",
    "blandChromatin",
    "normalNucleoli",
    "mitoses",
    "class"
]

data = pd.read_csv(
    "../data/dataset.csv",
    names=feature_names,
    na_values=["?"]
)

data.head()

# %% [markdown]
# ## Preprocessing steps
#
# 1. Data exploration
#     - [x] Binarize target variable
#     - [x] Missing values
#         - [ ] Replace by suitable estimate or
#         - [x] Discard
#     - [x] [Visualise the ratio of classes](#Visualise-the-ratio-of-classes)
#     - [x] Look at each feature
#         - Box plot
#         - Outliers
# 2. Normalization
#     - Might not be possible - look into
# 3. Skewing
#     - Check to see if the data are skewed towards some values
# 3. Correlation matrix
#     - Analyse the correlation between features and target; features and features
# 4. Feature selection
#     - Weakly correlated features can be discarded
#     - Duplicated features can be discarded
#     - PCA, RCA, ICA in a separate script    
#     - LDA for dimensionallity reduction

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 1. Data exploration

# %% [markdown] tags=[]
# ## Binarize the target variable
#
# The dataset is labelled as 2 indicating a benign tumour, and 4 indicating a malignant one. Firstly we will binarize this target label into 0 for benign and 1 for malignant.

# %%
data["class"] = (data["class"] > 2).astype('int')

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Consider missing values
#
# Next we can look for missing values in our data. The missing values in the data are labelled with `?`, which is passed to the csv reader to parse as `NaN`. When `NaN` is present in a column of `int64` the dtype is cast to float, so columns with `NaN` values will be of type `float64`.

# %%
data.dtypes

# %% [markdown]
# The `bareNuclei` column has been casted to `float64`, meaning there are some missing values in this column.

# %%
print(f'Values missing in bareNuclei? {any(data["bareNuclei"].isna())}') # Some values are NaN
np.count_nonzero(data.isna())

# %% [markdown]
# There are 16 instances of missing values. Rows containing them can now be removed.

# %%
data.dropna(axis='index', how='any', inplace=True)
data

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Visualise the ratio of classes
#
# The proportion of positive to negative class (malignant to benign cases) can be visualised using a historgram. Note since the target is now binarized the values will only be 0 or 1.

# %%
bin_vals, _, _ = plt.hist(data["class"], [-0.5, 0.5, 1.5], ec="k")
plt.xticks((0, 1), ("Benign", "Malignant"))
plt.show()

# %%
n_benign = bin_vals[0]
n_malig = bin_vals[1]

ratio = n_benign / (n_benign + n_malig) * 100
print(f"The dataset is comprised of {ratio:.2f}% beingn examples and {100-ratio:.2f}% malignant examples.")


# %% [markdown]
# There is an inbalance in the data; approximately 2/3 of the data are the negative class (benign). However it is not too bad/significant.

# %% [markdown] tags=[]
# ## Visualising the features

# %%
def draw_histogram(feature_index):
    title = re.sub('([A-Z]+)', r' \1', data.columns[feature_index]).lower()

    figure, (his, box) = plt.subplots(1, 2, figsize=(15, 8))
    figure.suptitle(f'Histogram and boxplot for {title}')
    
    # Plot a histogram with a nice title and labels
    his.hist(data.iloc[:, feature_index], ec="k")
    his.set_xlabel(data.columns[feature_index])
    his.set_ylabel("Count")

    box.boxplot(data.iloc[:, feature_index], notch=True)
    box.set_yticks(range(11))


# %%
widgets.interact(draw_histogram, feature_index=widgets.IntSlider(min=0, max=8))

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 4. Correlation Matrix

# %%
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # 5. Conclusion

# %%
