README
===========================
Inspired by [the work](https://stackoverflow.com/a/37366154/10404826) of Milo Spencer-Harper and Oli Blum, I created a simple Python module to visualize Multi-Layer Perceptron Neural Networks.

This module is able to:
1. Show the network architecture of the neural network (including the input layer, hidden layers, the output layer, the neurons in these layers, and the connections between neurons.)
2. Show the weights of the neural network using labels, colours and lines.

For more details, please refer to [my blog](http://www.jzliu.net/blog/simple-python-library-visualize-neural-network/).

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
X = training_set_inputs
y = training_set_outputs

classifier = MLPClassifier(hidden_layer_sizes=(4,), alpha=0.01, tol=0.001, random_state=1)
classifier.fit(X, y.ravel())

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
In all visualizations shown below, the weights are displayed using labels, different colors and linewidths. The organge color indicates a positive weight while the blue color indicates a negative weight. Only those weights that are greater than 0.5 or lesser than -0.5 are labeled.

ANN with 1 hidden layer (5 neurons in the input layer, 10 neurons in the hidden layer, and 1 neuron in the output layer)
![](/img/ANN_1.png "")

ANN with 2 hidden layers (5 neurons in the input layer, 15 neurons in the hidden layer 1, 10 neurons in the hidden layer 2, and 1 neuron in the output layer)
![](/img/ANN_2.png "")

ANN with 1 hidden layer (3 neurons in the input layer, 4 neurons in the hidden layer, and 1 neuron in the output layer)
![](/img/ANN_3.png)

ANN with 1 hidden layer without weights (3 neurons in the input layer, 4 neurons in the hidden layer, and 1 neuron in the output layer)
![](/img/ANN_4.png)
