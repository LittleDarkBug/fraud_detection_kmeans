import pandas as pd

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant numerical features for clustering.

    :param df: Full transaction DataFrame
    :return: DataFrame with selected features only
    """
    # Select relevant numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Filter out non-relevant columns (IDs, dates, etc.)
    exclude_patterns = ['id', 'ID', 'Id', 'date', 'Date', 'timestamp', 'time']
    selected_cols = [col for col in numeric_cols 
                    if not any(pattern in col for pattern in exclude_patterns)]
    
    selected_features = df[selected_cols]
    print(f"\nSélection de {len(selected_cols)} caractéristiques pour le clustering:")
    print(selected_cols)
    return selected_features