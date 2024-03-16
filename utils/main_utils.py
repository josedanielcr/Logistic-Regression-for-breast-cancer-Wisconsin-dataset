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


def visualize_2d_pca(x_train, y_train):
    """
    Apply PCA to reduce the features in x_train to two dimensions,
    and visualize the data in a 2D scatter plot along with y_train.

     Args:
        x_train (numpy array): Feature data.
        y_train (numpy array): Corresponding labels.
    """

    # Standardize the features
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)

    # Apply PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x_train_std)

    # Creating a DataFrame for the PCA results
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['principal component 1', 'principal component 2'])
    pca_df['diagnosis'] = y_train

    # Plotting the 2D data
    plt.figure(figsize=(8,6))
    targets = [0, 1]
    colors = ['b', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = pca_df['diagnosis'] == target
        plt.scatter(pca_df.loc[indicesToKeep, 'principal component 1'],
                    pca_df.loc[indicesToKeep, 'principal component 2'],
                    c=color, s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA of Breast Cancer Dataset')
    plt.legend(['Benign', 'Malignant'])
    plt.show()