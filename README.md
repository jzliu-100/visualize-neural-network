README
===========================
This is a Library to visualize Neural Networks based on the work by Milo Spencer-Harper and Oli Blum (https://stackoverflow.com/a/37366154/10404826).

****
	
|Author|Jianzheng Liu|
|---|---
|E-mail|jzliu.100@gmail.com


****
## Table of contents
* [Just give me the code](#just-give-me-the-code)
* [Gallery](#gallery)


Just give me the code
----------

### Visualize a Neural Network without weights
```Python
import VisualizeNN as VisNN
network=VisNN.DrawNN([3,4,1]])
network.draw()
```

### Visualize a Neural Network with weights
```Python
import VisualizeNN as VisNN
from sklearn.neural_network import MLPClassifier
import numpy as np

training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

######################
# Setup and train the Neural Network
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
```

Gallery
------
ANN with 1 hidden layer (5 neurons in the input layer, 10 neurons in the hidden layer, and 1 neuron in the output layer)
![](/img/ANN_1.png "")

ANN with 2 hidden layers (5 neurons in the input layer, 15 neurons in the hidden layer 1, 10 neurons in the hidden layer 2, and 1 neuron in the output layer)
![](/img/ANN_2.png "")

ANN with 1 hidden layer (3 neurons in the input layer, 4 neurons in the hidden layer, and 1 neuron in the output layer)
![](/img/ANN_3.png | width=400)

ANN with 1 hidden layer without weights (3 neurons in the input layer, 4 neurons in the hidden layer, and 1 neuron in the output layer)
![](/img/ANN_4.png | width=400)
