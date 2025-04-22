import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(data: np.ndarray, labels: np.ndarray, anomalies: np.ndarray, distances: np.ndarray, threshold: float):
    """
    Plot the clustered transactions and highlight anomalies.

    :param data: Normalized data
    :param labels: Labels assigned by KMeans
    :param anomalies: Indices of anomalies
    :param distances: Distances of each point to its cluster center
    :param threshold: Threshold used for anomaly detection
    """
    # Dimensional reduction with PCA for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster with a different color
    for cluster_label in np.unique(labels):
        # Normal points (non-anomalies)
        mask = (labels == cluster_label) & ~np.isin(np.arange(len(labels)), anomalies)
        cluster_points = reduced_data[mask]
        if len(cluster_points) > 0:
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1], 
                label=f'Cluster {cluster_label}',
                alpha=0.6
            )
    
    # Highlight anomalies in red
    if len(anomalies) > 0:
        plt.scatter(
            reduced_data[anomalies, 0], 
            reduced_data[anomalies, 1], 
            color='red', 
            label='Anomalies', 
            marker='x', 
            s=100
        )
    
    plt.title('Clustering des transactions avec mise en évidence des anomalies')
    plt.xlabel(f'Composante PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Composante PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Additional plot to visualize distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', 
                label=f'Seuil d\'anomalie: {threshold:.2f}')
    plt.title('Distribution des distances aux centres des clusters')
    plt.xlabel('Distance')
    plt.ylabel('Nombre de transactions')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()