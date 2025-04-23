import pandas as pd
from rich.console import Console

console = Console()

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from the given CSV file.

    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the transaction data
    """
    df = pd.read_csv(file_path)
    console.log(f"Dataset chargé avec {df.shape[0]} transactions et {df.shape[1]} colonnes")
    console.log("\nPremières lignes du dataset:")
    console.log(df.head())
    return df