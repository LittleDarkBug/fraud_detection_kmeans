import numpy as np

def analyze_cluster_specific_features(data, labels, feature_names, n_top_features=3):
    """
    Find which features best characterize each specific cluster versus all others.

    Parameters:
    - data: Normalized data used for clustering.
    - labels: Cluster labels assigned by the model.
    - feature_names: Names of the features in the data.
    - n_top_features: Number of top features to return for each cluster.
    
    Returns a dictionary mapping cluster IDs to their most distinctive features.
    """
    n_clusters = len(np.unique(labels))
    cluster_features = {}
    
    for cluster_id in range(n_clusters):
        in_cluster = (labels == cluster_id)
        cluster_importance = {}
        
        for feature_idx, feature in enumerate(feature_names):
            in_mean = np.mean(data[in_cluster, feature_idx])
            out_mean = np.mean(data[~in_cluster, feature_idx]) 
            pooled_std = np.sqrt((np.var(data[in_cluster, feature_idx]) + 
                                  np.var(data[~in_cluster, feature_idx])) / 2)
            
            if pooled_std > 0:
                effect_size = abs(in_mean - out_mean) / pooled_std
            else:
                effect_size = float('inf')
                
            cluster_importance[feature] = effect_size
            
        # Get top features for this cluster
        sorted_features = sorted(cluster_importance.items(), key=lambda x: x[1], reverse=True)
        cluster_features[cluster_id] = sorted_features[:n_top_features]
    
    return cluster_features