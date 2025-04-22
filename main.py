import numpy as np
import pandas as pd

from functions.load_data import load_data
from functions.select_features import select_features
from functions.normalize_features import normalize_features
from functions.apply_kmeans import apply_kmeans
from functions.detect_anomalies import detect_anomalies
from functions.plot_clusters import plot_clusters

# Import utility functions for console display
from functions.console_utils import (
    console, create_header, display_data_summary, format_cluster_table,
    format_anomalies_table, display_anomalies_stats, display_anomalies_distribution,
    create_progress_context, display_success_message, wait_for_user,
    display_step_header, display_cluster_analysis
)

def main():
    # Display header
    create_header()
    console.print("\n[bold]Ce programme s'execute etape par etape, pour chaque etape, appuyez sur [bold red]Entrée[/bold red] pour continuer.[/bold]")
    wait_for_user()
    
    # Step 1: Load data
    display_step_header(1, "Chargement des données")
    with create_progress_context() as progress:
        task1 = progress.add_task("[cyan]Chargement des données...", total=1)
        df = load_data("./data/transactions.csv")
        progress.update(task1, completed=1)
    
    # Display data summary
    display_data_summary(df)
    wait_for_user()
    
    # Step 2: Feature selection
    display_step_header(2, "Sélection des caractéristiques")
    with create_progress_context() as progress:
        task2 = progress.add_task("[cyan]Sélection des caractéristiques...", total=1)
        features_df = select_features(df)
        progress.update(task2, completed=1)
    wait_for_user()
    
    # Step 3: Feature normalization
    display_step_header(3, "Normalisation des caractéristiques")
    with create_progress_context() as progress:
        task3 = progress.add_task("[cyan]Normalisation des caractéristiques...", total=1)
        normalized_data = normalize_features(features_df)
        progress.update(task3, completed=1)
    wait_for_user()
    
    # Step 4: Apply K-means clustering
    display_step_header(4, "Application du clustering K-means")
    with create_progress_context() as progress:
        task4 = progress.add_task("[cyan]Application du clustering K-means...", total=1)
        kmeans_model, cluster_labels = apply_kmeans(normalized_data, n_clusters=6)
        progress.update(task4, completed=1)
    
    # Display cluster distribution
    console.print("\n[bold cyan]Distribution des clusters:[/bold cyan]")
    cluster_table = format_cluster_table(features_df, cluster_labels)
    console.print(cluster_table)
    wait_for_user()
    
    # Step 5: Anomaly detection
    display_step_header(5, "Détection des anomalies")
    with create_progress_context() as progress:
        task5 = progress.add_task("[cyan]Détection des anomalies...", total=1)
        
        # Calculate distances to determine appropriate threshold
        distances = np.zeros(normalized_data.shape[0])
        for i in range(normalized_data.shape[0]):
            cluster_label = kmeans_model.labels_[i]
            distances[i] = np.linalg.norm(normalized_data[i] - kmeans_model.cluster_centers_[cluster_label])
        
        # Define threshold as mean + 2.5 * standard deviation
        threshold = np.mean(distances) + 2.5 * np.std(distances)
        console.print(f"Seuil de distance pour les anomalies: [bold]{threshold:.4f}[/bold]")
        
        anomalies, distances = detect_anomalies(normalized_data, kmeans_model, threshold)
        progress.update(task5, completed=1)
    
    # Display anomaly information
    display_anomalies_stats(anomalies, normalized_data, threshold, distances)
    
    # Analyze anomaly distribution by cluster
    display_anomalies_distribution(cluster_labels, anomalies)
    
    # Display anomaly examples
    if len(anomalies) > 0:
        console.print(format_anomalies_table(features_df, cluster_labels, anomalies))
    wait_for_user()
    
    # Step 6: Results visualization
    display_step_header(6, "Visualization")
    console.print("[yellow]Préparation des visualisations...[/yellow]")
    # Calculation of average feature values per cluster
    display_cluster_analysis(features_df, cluster_labels)
    console.print("[italic]Note: Des fenetres peuvent s'ouvrir pour afficher les graphiques. Fermez-lesquand vous avez terminé.[/italic]")
    plot_clusters(normalized_data, cluster_labels, anomalies, distances, threshold)

    
    # Conclusion message
    display_success_message()

if __name__ == "__main__":
    main()