import numpy as np
import os
import platform
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

# Initialize Rich console
console = Console(force_terminal=True)

def create_header():
    """Creates a stylized header for the application."""
    console.print(Panel.fit(
        "[bold cyan]Détection d'anomalies dans les transactions bancaires[/bold cyan]\n"
        "[yellow]Algorithme K-means pour l'identification de comportements inhabituels[/yellow]",
        border_style="blue", padding=(1, 2)
    ))
    console.print("\n[bold]Ce programme s'execute etape par etape, pour chaque etape, appuyez sur [bold red]Entrée[/bold red] pour continuer.[/bold]")

def clear_console():
    """
    Clear the console screen based on the operating system.
    """
    command = 'cls' if platform.system().lower() == 'windows' else 'clear'
    os.system(command)

def wait_for_user():
    """
    Wait for user input to continue to the next step.
    """
    console.print("\n[bold cyan]Appuyez sur [bold]Entrée[/bold] pour continuer...[/bold cyan]")
    console.print("Pour quitter, appuyez sur [bold red]Ctrl + C[/bold red].")
    input()
    clear_console()

def display_step_header(step_number, step_title):
    """
    Displays a header for each execution step.
    
    Args:
        step_number: The step number
        step_title: The title of the step
    """
    console.print(Panel.fit(
        f"[bold cyan]Step {step_number}: {step_title}[/bold cyan]",
        border_style="blue"
    ))

def display_data_summary(df):
    """
    Displays a summary of the loaded dataset.
    
    Args:
        df: The pandas DataFrame containing the transaction data
    """
    console.print(Panel(
        f"Dataset chargé avec [bold]{df.shape[0]}[/bold] transactions et [bold]{df.shape[1]}[/bold] colonnes",
        border_style="green"
    ))
    
    table = Table(title="Aperçu des données", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    for col in df.columns:
        table.add_column(col)
    for _, row in df.head(5).iterrows():
        table.add_row(*[str(val) for val in row])
    console.print(table)
    console.print("")

def format_cluster_table(features_df, cluster_labels):
    """
    Creates a stylized table with cluster information.
    
    Args:
        features_df: DataFrame with the selected features
        cluster_labels: Array of cluster assignments
        
    Returns:
        Rich Table object containing cluster information
    """
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Cluster", style="dim")
    table.add_column("Taille", justify="right")
    table.add_column("% du total", justify="right")
    
    # Add columns for each feature
    for column in features_df.columns:
        table.add_column(column, justify="right")

    # Fill the table with data from each cluster
    for cluster in np.unique(cluster_labels):
        cluster_data = features_df.iloc[cluster_labels == cluster]
        cluster_size = len(cluster_data)
        percentage = f"{cluster_size / len(cluster_labels):.2%}"
        
        # Mean values for each feature
        values = [f"{cluster_data[col].mean():.4f}" for col in features_df.columns]
        
        # Add the row with a different style based on the cluster
        color = ["red", "green", "blue", "yellow", "cyan", "magenta"][cluster % 6]
        table.add_row(
            f"[bold {color}]{cluster}[/bold {color}]",
            f"{cluster_size}",
            percentage,
            *values,
            style=f"dim {color}"
        )
        
    return table

def format_anomalies_table(features_df, cluster_labels, anomalies, sample_size=6):
    """
    Creates a stylized table with anomaly samples, ensuring at least one from each cluster if possible.
    
    Args:
        features_df: DataFrame with the selected features
        cluster_labels: Array of cluster assignments
        anomalies: Array of indices of anomalous transactions
        sample_size: Number of anomaly examples to show
        
    Returns:
        Rich Panel containing the anomalies table
    """
    if len(anomalies) == 0:
        return Panel("Aucune anomalie détectée.", border_style="red")
        
    table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
    table.add_column("ID", style="dim")
    table.add_column("Cluster", justify="center")
    
    for column in features_df.columns:
        table.add_column(column, justify="right")
    
    table.add_column("Déviation moyenne", justify="right")
    
    clusters_with_anomalies = {}
    for idx in anomalies:
        cluster = cluster_labels[idx]
        if cluster not in clusters_with_anomalies:
            clusters_with_anomalies[cluster] = []
        clusters_with_anomalies[cluster].append(idx)
    
    # Select one anomaly from each cluster first
    selected_indices = []
    for cluster in sorted(clusters_with_anomalies.keys()):
        selected_indices.append(np.random.choice(clusters_with_anomalies[cluster], 1)[0])
    
    # If we need more samples, select randomly from the remaining anomalies
    remaining_count = sample_size - len(selected_indices)
    if remaining_count > 0:
        remaining_anomalies = [idx for idx in anomalies if idx not in selected_indices]
        
        if len(remaining_anomalies) > 0:
            additional_samples = np.random.choice(
                remaining_anomalies, 
                min(remaining_count, len(remaining_anomalies)), 
                replace=False
            )
            selected_indices.extend(additional_samples)
    

    np.random.shuffle(selected_indices)
    

    for idx in selected_indices:
        cluster = cluster_labels[idx]
        transaction = features_df.iloc[idx]
        
        # Calculate deviations from the cluster mean
        cluster_mean = features_df.iloc[cluster_labels == cluster].mean()
        deviations = []
        values = []
        
        for col in features_df.columns:
            value = transaction[col]
            mean = cluster_mean[col]
            deviation = ((value - mean) / mean * 100) if mean != 0 else float('inf')
            deviations.append(deviation)
            
     
            if abs(deviation) > 100:
                values.append(f"[bold red]{value:.2f}[/bold red]")
            elif abs(deviation) > 50:
                values.append(f"[orange3]{value:.2f}[/orange3]")
            else:
                values.append(f"{value:.2f}")
        
        avg_deviation = sum(abs(d) for d in deviations if d != float('inf')) / len([d for d in deviations if d != float('inf')])        
  
        table.add_row(
            f"{idx}",
            f"[bold]C{cluster}[/bold]",
            *values,
            f"[bold]{avg_deviation:.2f}%[/bold]"
        )
    
    return Panel(table, title="[bold red]Échantillons d'anomalies[/bold red]", border_style="red")

def display_anomalies_stats(anomalies, normalized_data, threshold, distances):
    """
    Displays statistics about the detected anomalies.
    
    Args:
        anomalies: Array of indices of anomalous transactions
        normalized_data: Normalized feature array
        threshold: Distance threshold used for anomaly detection
        distances: Array of distances from each point to its cluster center
    """
    console.print(Panel(
        f"[bold]Détection d'anomalies[/bold]\n\n"
        f"Seuil de distance: [bold]{threshold:.4f}[/bold]\n"
        f"Nombre d'anomalies: [bold red]{len(anomalies)}[/bold red] ([bold]{len(anomalies)/len(normalized_data):.2%}[/bold] des transactions)\n"
        f"Distance moyenne: [bold]{np.mean(distances):.4f}[/bold]\n"
        f"Distance maximale: [bold]{np.max(distances):.4f}[/bold]",
        border_style="yellow", title="Statistiques des anomalies"
    ))

def display_anomalies_distribution(cluster_labels, anomalies):
    """
    Displays the distribution of anomalies by cluster.
    
    Args:
        cluster_labels: Array of cluster assignments
        anomalies: Array of indices of anomalous transactions
    """
    if len(anomalies) > 0:
        anomaly_clusters = cluster_labels[anomalies]
        unique, counts = np.unique(anomaly_clusters, return_counts=True)
        
        anomaly_table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
        anomaly_table.add_column("Cluster")
        anomaly_table.add_column("Nb anomalies", justify="right")
        anomaly_table.add_column("% des anomalies", justify="right")
        anomaly_table.add_column("% du cluster", justify="right")
        
        for label, count in zip(unique, counts):
            cluster_size = np.sum(cluster_labels == label)
            anomaly_table.add_row(
                f"Cluster {label}",
                f"{count}",
                f"{count/len(anomalies):.2%}",
                f"{count/cluster_size:.2%}"
            )
        
        console.print(Panel(anomaly_table, title="[bold]Distribution des anomalies par cluster[/bold]", border_style="yellow"))

def create_progress_context():
    """
    Creates and returns a progress bar context.
    
    Returns:
        Rich Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
    )

def display_success_message():
    """Displays a success message at the end of the process."""
    console.print(
        Panel.fit(
            "[bold green]Détection d'anomalies terminée avec succès![/bold green]\n"
            "Veuillez consulter le rapport pour plus de détails.",
            title="Conclusion",
            border_style="green"
        )
    )

def display_cluster_analysis(features_df, cluster_labels):
    """
    Displays an analysis of the features for each cluster.
    
    Args:
        features_df: DataFrame with the selected features
        cluster_labels: Array of cluster assignments
    """
    console.print("\n[bold]Caractéristiques des clusters (valeurs moyennes):[/bold]")
    for cluster in np.unique(cluster_labels):
        cluster_data = features_df.iloc[cluster_labels == cluster]
        console.print(f"\n[bold]Cluster {cluster}[/bold] ({len(cluster_data)} transactions):")
        for column in features_df.columns:
            console.print(f"  {column}: {cluster_data[column].mean():.4f}")


def update_and_display_anomaly_stats(cluster_info, kmeans_model, normalized_data, anomalies):
    """ Update cluster info with anomaly statistics and build display table
    Args:
        cluster_info: List of dictionaries with cluster information
        kmeans_model: Trained KMeans model
        normalized_data: Normalized feature array
        anomalies: Array of indices of anomalous transactions
    Returns:
        cluster_info: Updated list of dictionaries with cluster information
        threshold_table: Rich Table object containing updated cluster information
    """

    # Update anomaly stats for each cluster
    for info in cluster_info:
        cluster_id = info['id']
        cluster_mask = (kmeans_model.labels_ == cluster_id)
        anomaly_mask = np.isin(np.arange(len(normalized_data)), anomalies)
        cluster_anomalies = np.sum(cluster_mask & anomaly_mask)
        info['anomalies'] = cluster_anomalies
        info['anomalies_percent'] = cluster_anomalies / info['size'] if info['size'] > 0 else 0
    
    # Build display table
    threshold_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    threshold_table.add_column("Cluster", style="dim")
    threshold_table.add_column("Taille", justify="right")
    threshold_table.add_column("Seuil", justify="right")
    threshold_table.add_column("Anomalies", justify="right")
    threshold_table.add_column("% Anomalies", justify="right")
    
    for info in cluster_info:
        threshold_table.add_row(
            f"{info['id']}", 
            f"{info['size']}", 
            f"{info['threshold']:.4f}", 
            f"{info['anomalies']}", 
            f"{info['anomalies_percent']:.2%}"
        )
    
    return cluster_info, threshold_table


def format_threshold_table(cluster_info):
    """
    Create a formatted table displaying anomaly thresholds for each cluster.
    
    Parameters:
        cluster_info (list): List of dictionaries containing cluster information
        
    Returns:
        Table: Rich Table object containing formatted threshold information
    """
    threshold_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    threshold_table.add_column("Cluster", style="dim")
    threshold_table.add_column("Taille", justify="right")
    threshold_table.add_column("Q1", justify="right")
    threshold_table.add_column("Q3", justify="right")
    threshold_table.add_column("IQR", justify="right")
    threshold_table.add_column("Seuil", justify="right")
    
    for info in cluster_info:
        threshold_table.add_row(
            f"{info['id']}", 
            f"{info['size']}", 
            f"{info['q1']:.4f}", 
            f"{info['q3']:.4f}", 
            f"{info['iqr']:.4f}", 
            f"{info['threshold']:.4f}"
        )
    
    return threshold_table