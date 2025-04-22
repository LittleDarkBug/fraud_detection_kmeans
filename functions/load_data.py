import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from the given CSV file.

    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the transaction data
    """
    df = pd.read_csv(file_path)
    print(f"Dataset chargé avec {df.shape[0]} transactions et {df.shape[1]} colonnes")
    print("\nPremières lignes du dataset:")
    print(df.head())
    return df