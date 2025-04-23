from functions.load_data import load_data
from functions.cluster_analysis import  analyze_cluster_specific_features
from functions.select_features import select_features
from functions.normalize_features import normalize_features
from functions.apply_kmeans import apply_kmeans
from functions.detect_anomalies import detect_anomalies
from functions.plot_clusters import  plot_clusters, plot_distinctive_features
from functions.threshold_utils import calculate_thresholds


# Utility for console display
from functions.console_utils import (
    console, create_header, format_cluster_table,
    format_anomalies_table, display_anomalies_distribution,
    create_progress_context, display_success_message,
    format_threshold_table, update_and_display_anomaly_stats, wait_for_user,
    display_step_header, display_cluster_analysis
)


def main():
    create_header()
    wait_for_user()
    
    # Step 1: Load data
    display_step_header(1, "Chargement des données")
    with create_progress_context() as progress:
        task = progress.add_task("[cyan]Chargement des données...", total=1)
        df = load_data("./data/transactions.csv")
        progress.update(task, completed=1)
    wait_for_user()
    
    # Step 2: Feature selection
    display_step_header(2, "Sélection des caractéristiques")
    with create_progress_context() as progress:
        task = progress.add_task("[cyan]Sélection des caractéristiques...", total=1)
        features_df = select_features(df)
        progress.update(task, completed=1)
    wait_for_user()
    
    # Step 3: Feature normalization
    display_step_header(3, "Normalisation des caractéristiques")
    with create_progress_context() as progress:
        task = progress.add_task("[cyan]Normalisation des caractéristiques...", total=1)
        normalized_data = normalize_features(features_df)
        progress.update(task, completed=1)
    wait_for_user()
    
    # Step 4: K-means clustering
    display_step_header(4, "Application du clustering K-means")
    with create_progress_context() as progress:
        task = progress.add_task("[cyan]Application du clustering K-means...", total=1)
        kmeans_model, cluster_labels = apply_kmeans(normalized_data, n_clusters=6)
        progress.update(task, completed=1)
    
    # Cluster distribution
    console.print("\n[bold cyan]Distribution des clusters:[/bold cyan]")
    console.print(format_cluster_table(features_df, cluster_labels))
    wait_for_user()

    
    # Step 5: Anomaly detection
    display_step_header(5, "Détection des anomalies avec seuil dynamique par cluster")
    with create_progress_context() as progress:
        task = progress.add_task("[cyan]Calcul des seuils d'anomalies par cluster...", total=1)
        thresholds, cluster_info = calculate_thresholds(normalized_data, kmeans_model)
        progress.update(task, completed=1)

        console.print("\n[bold cyan]Seuils d'anomalies par cluster (IQR):[/bold cyan]")
        console.print(format_threshold_table(cluster_info))
        
        # Detect anomalies
        task = progress.add_task("[cyan]Détection des anomalies avec les seuils calculés...", total=1)
        anomalies, distances = detect_anomalies(normalized_data, kmeans_model, thresholds)
        progress.update(task, completed=1)
        
        # Update and display anomaly statistics
        cluster_info, anomaly_table = update_and_display_anomaly_stats(
            cluster_info, kmeans_model, normalized_data, anomalies
        )
        console.print("\n[bold cyan]Détection d'anomalies par cluster:[/bold cyan]")
        console.print(anomaly_table)
        
        # Anomaly distribution and examples
        display_anomalies_distribution(kmeans_model.labels_, anomalies)
        if len(anomalies) > 0:
            console.print(format_anomalies_table(features_df, kmeans_model.labels_, anomalies))
        wait_for_user()
    
    # Step 6: Visualization
    display_step_header(6, "Visualization")
    console.print("[yellow]Préparation des visualisations...[/yellow]")
    display_cluster_analysis(features_df, cluster_labels)
    console.print("[italic]Note: Des fenetres peuvent s'ouvrir pour afficher les graphiques. Fermez-les quand vous avez terminé.[/italic]")
    plot_clusters(normalized_data, cluster_labels, anomalies, distances)
    cluster_features = analyze_cluster_specific_features(normalized_data, cluster_labels, features_df.columns.tolist())
    plot_distinctive_features(cluster_features)
    display_success_message()

if __name__ == "__main__":
    main()