import VisualizeNN as VisNN
import os
import numpy as np

training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

######################
# Setup and train the Neural Network
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X = training_set_inputs
y = training_set_outputs

classifier = MLPClassifier(hidden_layer_sizes=(4,), alpha=0.01, tol=0.001, random_state=1)
classifier.fit(X, y.ravel())

######################
# Visualize the Neural Network
network_structure = np.hstack(([X.shape[1]], np.asarray(classifier.hidden_layer_sizes), [y.shape[1]]))
# Draw the Neural Network with weights
network=VisNN.DrawNN(network_structure, classifier.coefs_)
network.draw()

# Draw the Neural Network without weights
network=VisNN.DrawNN(network_structure)
network.draw()