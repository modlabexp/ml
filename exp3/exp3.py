import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

data = pd.read_csv("diabetes.csv")

x = data.iloc[:, :-1]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

n_clusters = 2

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(x_scaled)

kmeans_score = silhouette_score(
    x_scaled,
    kmeans_labels
)

print("K-Means Silhouette Score:", round(kmeans_score, 3))

gmm = GaussianMixture(
    n_components=n_clusters,
    random_state=42
)

gmm_labels = gmm.fit_predict(x_scaled)

gmm_score = silhouette_score(
    x_scaled,
    gmm_labels
)

print("EM (GMM) Silhouette Score:", round(gmm_score, 3))
