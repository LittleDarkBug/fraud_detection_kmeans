import numpy as np
from sklearn.cluster import KMeans

def detect_anomalies(data: np.ndarray, model: KMeans, threshold: float | dict) -> tuple:
    """
    Detect transactions that are far from their cluster center.

    :param data: Normalized data
    :param model: Trained KMeans model
    :param threshold: Dictionary of thresholds per cluster or a single global threshold value
    :return: Tuple (Array of indices of anomalous transactions, All distances)
    """
    distances = np.zeros(data.shape[0])
    
    # Determine if we're using dynamic thresholds or a global threshold
    use_dynamic_thresholds = isinstance(threshold, dict)
    
    # Distance to center
    for i in range(data.shape[0]):
        cluster_label = model.labels_[i]
        cluster_center = model.cluster_centers_[cluster_label]
        distances[i] = np.linalg.norm(data[i] - cluster_center)
    
    is_anomaly = np.zeros(data.shape[0], dtype=bool)
    
    # Use  thresholds
    if use_dynamic_thresholds:
        for i in range(data.shape[0]):
            cluster_label = model.labels_[i]
            if distances[i] > threshold[cluster_label]:
                is_anomaly[i] = True
    else:
        is_anomaly = distances > threshold
    
    anomalies = np.where(is_anomaly)[0]
    return anomalies, distances