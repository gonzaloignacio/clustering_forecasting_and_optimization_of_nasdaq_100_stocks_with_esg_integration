# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the input file and select the necessary variables
cluster_input_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_input.csv",
                               index_col = [0])
cluster_input_df = cluster_input_df.iloc[:, 1:]


# Import scaler to standardize data and fit it to the input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(cluster_input_df)

# Import model and comparison metric
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_samples, silhouette_score

# Set seed to ensure reproducibility
seed = 0
np.random.seed(seed)

# Initialize the empty list for the full dataset
all_sil_score = []

# Obtain the best number of clusters using the silhouette score
for i in range (2, 11):
    cluster = KMeansConstrained(n_clusters = i,
                                size_min = 5,
                                size_max = 50,
                                random_state = seed)
    cluster.fit(X_cluster_scaled)
    score = silhouette_score(X_cluster_scaled, 
                             cluster.predict(X_cluster_scaled))
    all_sil_score.append(score)

# Fix the number of clusters and run the model
all_best_n_clusters = pd.Series(all_sil_score).idxmax() + 2

all_best_cluster = KMeansConstrained(n_clusters = all_best_n_clusters,
                                 size_min = 5,
                                 size_max = 50,
                                 random_state = seed)
all_best_cluster.fit(X_cluster_scaled)

# Get labels, silhouette individual values and average values
all_cluster_labels = all_best_cluster.fit_predict(X_cluster_scaled)
all_silhouette_vals = silhouette_samples(X_cluster_scaled, all_cluster_labels)
all_avg_score = silhouette_score(X_cluster_scaled, all_cluster_labels)

# Get the information in a dataframe
all_sil_df = pd.DataFrame({
    "Cluster": all_cluster_labels,
    "Silhouette": all_silhouette_vals
})

# Plot the values
sns.barplot(data = all_sil_df, x = "Cluster", y = "Silhouette", estimator=np.mean, ci=None)
plt.title("Average Silhouette Score per Cluster")
plt.axhline(all_avg_score, color = "red")
plt.show()



###############################################################################



# Import the PCA function
from sklearn.decomposition import PCA

# Fit the PCA to the scaled data
pca = PCA(random_state = seed)
pca.fit(X_cluster_scaled)

# Get the number of parameters that capture at least 80% of the variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()
pca_idx = next(i for i, val in enumerate(cumulative_variance) if val > 0.8) + 1

# Plot the cumulative variance
sns.barplot(x = range(1, 11), y = cumulative_variance)
plt.axhline(0.8, color = "red")
plt.xlabel("Number of PCA components")
plt.ylabel("Proportion of variance")
plt.title("Proportion of variance by Number of PCA components")
plt.show()

# Fit the PCA model with the optimal parameters and transform the input data
best_pca = PCA(n_components = pca_idx, random_state = seed)
X_pca_scaled = best_pca.fit_transform(X_cluster_scaled)

# Initialize the silhouette score list
pca_sil_score = []

# Obtain the optimal number of clusters for these variables
for i in range (2, 11):
    cluster = KMeansConstrained(n_clusters = i,
                                size_min = 5,
                                size_max = 50,
                                random_state = seed)
    cluster.fit(X_pca_scaled)
    score = silhouette_score(X_pca_scaled, 
                             cluster.predict(X_pca_scaled))
    pca_sil_score.append(score)
    
pca_best_n_clusters = pd.Series(pca_sil_score).idxmax() + 2

#Fit the odel with the best number of clusters
pca_best_cluster = KMeansConstrained(n_clusters = pca_best_n_clusters, 
                                     size_min = 5,
                                     size_max = 50,
                                     random_state = seed)
pca_best_cluster.fit(X_pca_scaled)

# Get labels, silhouette individual values and average values
pca_cluster_labels = pca_best_cluster.fit_predict(X_pca_scaled)
pca_silhouette_vals = silhouette_samples(X_pca_scaled, pca_cluster_labels)
pca_avg_score = silhouette_score(X_pca_scaled, pca_cluster_labels)

# Get the information in a dataframe
pca_sil_df = pd.DataFrame({
    "Cluster": pca_cluster_labels,
    "Silhouette": pca_silhouette_vals
})

# Plot the values
sns.barplot(data = pca_sil_df, x = "Cluster", y = "Silhouette", estimator=np.mean, ci=None)
plt.title("Average Silhouette Score per Cluster")
plt.axhline(pca_avg_score, color = "red")
plt.show()

###############################################################################



# Get the market columns and scale them
X_cluster_mkt = cluster_input_df.iloc[:, 7 : ]
scaler_mkt = StandardScaler()
X_cluster_scaled_mkt = scaler_mkt.fit_transform(X_cluster_mkt)

# Initialize the silhouette score list
mkt_sil_score= []

# Get the optimal number of clusters and store the value
for i in range (2, 11):
    cluster = KMeansConstrained(n_clusters = i,
                     size_min = 5,
                     size_max = 50,
                     random_state = seed)
    cluster.fit(X_cluster_scaled_mkt)
    score = silhouette_score(X_cluster_scaled_mkt, 
                             cluster.predict(X_cluster_scaled_mkt))
    mkt_sil_score.append(score)
    

mkt_best_n_clusters = pd.Series(mkt_sil_score).idxmax() + 2

# Run the model with the best number of clusters
mkt_best_cluster = KMeansConstrained(n_clusters = mkt_best_n_clusters,
                                     size_min = 5,
                                     size_max = 50,
                                     random_state = seed)

mkt_best_cluster.fit(X_cluster_scaled_mkt)

# Get labels, silhouette individual values and average values
mkt_cluster_labels = mkt_best_cluster.fit_predict(X_cluster_scaled_mkt)
mkt_silhouette_vals = silhouette_samples(X_cluster_scaled_mkt, mkt_cluster_labels)
mkt_avg_score = silhouette_score(X_cluster_scaled_mkt, mkt_cluster_labels)

# Get the information in a dataframe
mkt_sil_df = pd.DataFrame({
    "Cluster": mkt_cluster_labels,
    "Silhouette": mkt_silhouette_vals
})

# Plot the values
sns.barplot(data = mkt_sil_df, x = "Cluster", y = "Silhouette", estimator=np.mean, ci=None)
plt.title("Average Silhouette Score per Cluster")
plt.axhline(mkt_avg_score, color = "red")
plt.show()



###############################################################################



# Get the fundamental analysis indicators
X_cluster_fai = cluster_input_df.iloc[:, 0:7]
scaler_fai = StandardScaler()
X_cluster_scaled_fai = scaler_fai.fit_transform(X_cluster_fai)

# Initialize the silhouette score list
fai_sil_score = []

# Get the optimal number of clusters and store the value
for i in range (2, 11):
    cluster = KMeansConstrained(n_clusters = i,
                                size_min = 5,
                                size_max = 50,
                                random_state = seed)
    cluster.fit(X_cluster_scaled_fai)
    score = silhouette_score(X_cluster_scaled_fai, 
                             cluster.predict(X_cluster_scaled_fai))
    fai_sil_score.append(score)

fai_best_n_clusters = pd.Series(fai_sil_score).idxmax() + 2

# Run the model with the best number of clusters
fai_best_cluster = KMeansConstrained(n_clusters = fai_best_n_clusters,
                                     size_min = 5,
                                     size_max = 50,
                                     random_state = seed)

fai_best_cluster.fit(X_cluster_scaled_fai)

# Get labels, silhouette individual values and average values
fai_cluster_labels = fai_best_cluster.fit_predict(X_cluster_scaled_fai)
fai_silhouette_vals = silhouette_samples(X_cluster_scaled_fai, fai_cluster_labels)
fai_avg_score = silhouette_score(X_cluster_scaled_fai, fai_cluster_labels)

# Get the information in a dataframe
fai_sil_df = pd.DataFrame({
    "Cluster": fai_cluster_labels,
    "Silhouette": fai_silhouette_vals
})

# Plot the values
sns.barplot(data = fai_sil_df, x = "Cluster", y = "Silhouette", estimator=np.mean, ci=None)
plt.title("Average Silhouette Score per Cluster")
plt.axhline(fai_avg_score, color = "red")
plt.show()



###############################################################################




# Save the label from the best model
cluster_input_df["Cluster"] = mkt_best_cluster.labels_

# Plot clusters volatility vs return
sns.scatterplot(data = cluster_input_df, 
                x = "Historic Volatility", 
                y = "Historic Return",
                hue = "Cluster")
plt.xlabel("Historic Volatility")
plt.ylabel("Historic Return")
plt.title("Clusters by Return vs. Volatility")
plt.show()

# Plot clusters volume vs return
sns.scatterplot(data = cluster_input_df, 
                x = "Historic Return", 
                y = "Historic Volume",
                hue = "Cluster")
plt.xlabel("Historic Return")
plt.ylabel("Historic Volume (log scale)")
plt.title("Clusters by Volume vs. Return")
plt.yscale("log")
plt.show()

# Plot clusters volatility vs volume
sns.scatterplot(data = cluster_input_df, 
                x = "Historic Volatility", 
                y = "Historic Volume",
                hue = "Cluster")
plt.xlabel("Historic Volatility")
plt.ylabel("Historic Volume ( log scale)")
plt.title("Clusters by Volume vs. Volatility")
plt.yscale("log")
plt.show()

# Get the centroids measurements
centroids_df = cluster_input_df.groupby("Cluster").mean()

# Save a csv with the results
cluster_input_df.to_csv(r"C:\Users\gonza\Documents\cluster_labels.csv")
