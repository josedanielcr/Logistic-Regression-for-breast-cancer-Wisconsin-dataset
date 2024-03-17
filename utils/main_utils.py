import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path):
    """
    Load the dataset from the given file path and return the feature data (X) and labels (y).

     Args:
        file_path (str): The path to the dataset file.

    Returns:
        tuple: A tuple containing two numpy arrays, x_data (features) and y_data (labels).
    """

    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate features and labels
    x_data = data.drop('diagnosis', axis=1).values
    y_data = data['diagnosis'].values

    return x_data, y_data

