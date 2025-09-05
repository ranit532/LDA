import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_data(n_samples=150, n_features=2, n_classes=3, random_state=42):
    """
    Generates a synthetic dataset for classification.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The number of features.
        n_classes (int): The number of classes.
        random_state (int): The random state for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["target"] = y
    return df

if __name__ == "__main__":
    data = generate_data()
    data.to_csv("c:/Users/2185206/LDA/data/lda_data.csv", index=False)
    print("Data generated and saved to data/lda_data.csv")