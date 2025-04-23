from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(data: np.ndarray, labels: np.ndarray, anomalies: np.ndarray, distances: np.ndarray):
    """
    Plot the clustered transactions and highlight anomalies.

    :param data: Normalized data
    :param labels: Labels assigned by KMeans
    :param anomalies: Indices of anomalies
    :param distances: Distances of each point to its cluster center
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    plt.figure(figsize=(12, 8))
    
    for cluster_label in np.unique(labels):
 
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
    plt.savefig("./report/clustering_plot.png")

    # Histograms of distances for each cluster
    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(nrows=(n_clusters+1)//2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, cluster_label in enumerate(np.unique(labels)):
        if i < len(axes):
            ax = axes[i]
            cluster_mask = (labels == cluster_label)
            cluster_distances = distances[cluster_mask]
            
            # Use IQR to set threshold for this cluster
            q1 = np.percentile(cluster_distances, 25)
            q3 = np.percentile(cluster_distances, 75)
            iqr = q3 - q1
            cluster_threshold = q3 + 1.5 * iqr
            
            ax.hist(cluster_distances, bins=30, alpha=0.7)
            ax.axvline(x=cluster_threshold, color='r', linestyle='--', 
                      label=f'Seuil: {cluster_threshold:.2f}')
            ax.set_title(f'Cluster {cluster_label} (n={sum(cluster_mask)})')
            ax.set_xlabel('Distance au centre')
            ax.set_ylabel('Nombre de points')
            ax.grid(alpha=0.3)
            ax.legend()
    
    # Remove any unused subplots
    for i in range(n_clusters, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig("./report/clustering_analysis.png")
    plt.show()



def plot_distance_distributions(model, thresholds_iqr, distances, thresholds_adaptive=None):
    """
    Plot distance distributions for each cluster with thresholds.
    
    Parameters:
        model: Trained KMeans model
        thresholds_iqr: Dictionary mapping cluster ID to IQR threshold
        thresholds_adaptive: Dictionary mapping cluster ID to adaptive threshold (optional)
        distances: Pre-calculated distances (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    centers = model.cluster_centers_
    labels = model.labels_
    n_clusters = len(centers)
    
    n_rows = (n_clusters + 1) // 2
    fig = plt.figure(figsize=(14, n_rows * 4))
    gs = GridSpec(n_rows, 2, figure=fig)
    
    # Create histograms for each cluster
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_distances = distances[cluster_mask]
        
        if len(cluster_distances) == 0:
            continue
            
        ax = fig.add_subplot(gs[cluster_id // 2, cluster_id % 2])
        bins = np.linspace(0, max(cluster_distances) * 1.1, 30)
        ax.hist(cluster_distances, bins=bins, alpha=0.7, color='skyblue')
        
        # Add threshold lines
        iqr_threshold = thresholds_iqr[cluster_id] if isinstance(thresholds_iqr, dict) else thresholds_iqr
        ax.axvline(x=iqr_threshold, color='red', linestyle='--', 
                   label='IQR Threshold', linewidth=2)
        
        # Add adaptive threshold if provided
        if thresholds_adaptive is not None:
            adaptive_threshold = thresholds_adaptive[cluster_id]
            ax.axvline(x=adaptive_threshold, color='green', linestyle='-.', 
                       label='Adaptive Threshold', linewidth=2)
        
        median = np.median(cluster_distances)
        ax.axvline(x=median, color='navy', linestyle='-', 
                   label='Median', linewidth=1)
        
        # Mark anomalies
        ax.set_title(f'Cluster {cluster_id} Distance Distribution', fontsize=12)
        ax.set_xlabel('Distance to Cluster Center', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        
        # Only include legend on the first plot
        if cluster_id == 0:
            ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig("./report/distance_distributions.png")
    plt.show()


def plot_distinctive_features(cluster_features, n_features=3):
    """
    Create a visual representation of the distinctive features for each cluster.
    
    Parameters:
        cluster_features: Dictionary mapping cluster IDs to lists of (feature, score) tuples
        n_features: Number of top features to display per cluster (default: 3)
    """
    
    n_clusters = len(cluster_features)
    figsize=(12, 10)
    n_rows = (n_clusters + 1) // 2  
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, 2, figure=fig)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Create a subplot for each cluster
    for i, (cluster_id, features) in enumerate(sorted(cluster_features.items())):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        top_features = features[:min(n_features, len(features))]
        feature_names = [f[0] for f in top_features]
        scores = [f[1] for f in top_features]
        
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, scores, align='center', color=colors[i], alpha=0.7)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', ha='left', va='center')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis() 
        ax.set_xlabel('Score d\'importance')
        ax.set_title(f'Cluster {cluster_id} - Caractéristiques distinctives')
        ax.set_facecolor((*colors[i][:3], 0.05))
    
    plt.tight_layout()
    plt.savefig("./report/distinctive_features.png")
    plt.show()