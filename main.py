from utils import main_utils

# loads the training data
x_train, y_train = main_utils.load_dataset("training_data/selected_data.csv")

# data visualization
main_utils.visualize_2d_pca(x_train, y_train)