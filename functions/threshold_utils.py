import numpy as np
from sklearn.cluster import KMeans

def calculate_thresholds(data: np.ndarray, model: KMeans, iqr_multiplier: float = 1.5) -> dict:
    """
    Calculate anomaly thresholds for each cluster using Tukey (IQR).
    
    :param data: Normalized data
    :param model: Trained KMeans model
    :param iqr_multiplier: Multiplier for IQR (default 1.5)
    :return: Dictionary with cluster IDs as keys and thresholds as values
    """
    thresholds = {}
    cluster_info = []
    
    for cluster_id in range(model.n_clusters):
        # Get points in this cluster
        cluster_mask = (model.labels_ == cluster_id)
        cluster_points = data[cluster_mask]
        
        if len(cluster_points) == 0:
            thresholds[cluster_id] = float('inf')  # No points in this cluster
            continue
        
        # Calculate distances to cluster center
        cluster_distances = np.zeros(len(cluster_points))
        for i, point_idx in enumerate(np.where(cluster_mask)[0]):
            cluster_distances[i] = np.linalg.norm(data[point_idx] - model.cluster_centers_[cluster_id])
        
        # Apply IQR
        q1 = np.percentile(cluster_distances, 25)
        q3 = np.percentile(cluster_distances, 75)
        iqr = q3 - q1
        threshold = q3 + iqr_multiplier * iqr
        
        thresholds[cluster_id] = threshold
        
        cluster_info.append({
            'id': cluster_id,
            'size': len(cluster_points),
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'threshold': threshold,
        })
    
    return thresholds, cluster_info