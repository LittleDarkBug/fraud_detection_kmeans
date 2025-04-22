import numpy as np
from sklearn.cluster import KMeans

def apply_kmeans(data: np.ndarray, n_clusters: int = 6) -> tuple:
    """
    Apply KMeans clustering to the normalized transaction data.

    :param data: Normalized feature array
    :param n_clusters: Number of clusters
    :return: Tuple (trained model, labels)
    """

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    return kmeans, labels