import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_features(df: pd.DataFrame) -> np.ndarray:
    """
    Normalize the selected features using StandardScaler.

    :param df: DataFrame of selected features
    :return: Normalized NumPy array of features
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    print("\nNormalisation des données effectuée.")
    return normalized_data