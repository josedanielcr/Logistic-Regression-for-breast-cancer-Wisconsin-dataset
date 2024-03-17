import numpy as np

from models import logistic_regression
from utils import main_utils

# loads the training data
x_train, y_train = main_utils.load_dataset("training_data/selected_data.csv")

# run logistic regression
w, b = logistic_regression.execute(x_train, y_train)

p = logistic_regression.predict(x_train, w, b)

print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))