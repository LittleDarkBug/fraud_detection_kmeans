import numpy as np
from sklearn.cluster import KMeans

def detect_anomalies(data: np.ndarray, model: KMeans, threshold: float) -> tuple:
    """
    Detect transactions that are far from their cluster center.

    :param data: Normalized data
    :param model: Trained KMeans model
    :param threshold: Distance threshold
    :return: Tuple (Array of indices of anomalous transactions, All distances)
    """
    # Calculate distances between each point and its cluster center
    distances = np.zeros(data.shape[0])
    
    for i in range(data.shape[0]):
        cluster_label = model.labels_[i]
        cluster_center = model.cluster_centers_[cluster_label]
        distances[i] = np.linalg.norm(data[i] - cluster_center)
    
    # Points with distances above the threshold are anomalies
    anomalies = np.where(distances > threshold)[0]
    
    return anomalies, distances