from dataclasses import dataclass
from typing import Callable
import random 

import numpy as np 

# TODO: use python typing to do actual types 
# TODO: implement momentum
# TODO: make output look nicer like keras https://www.tensorflow.org/tutorials/keras/classification
# TODO: make output optional 
# TODO: make shape a prop so when u set it it reinitializes stuff
# TODO: different activations, ReLU, LReLU

# assorted functions
def sigmoid(x: np.ndarray): 
    """Sigmoid"""
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x: np.ndarray): 
    """Derivative of Sigmoid"""
    return sigmoid(x)*(1-sigmoid(x))

def ReLU(x: np.ndarray): 
    """Rectified Linear Unit"""
    return x * (x > 0)

def d_ReLU(x: np.ndarray): 
    """Derivative of Rectified Linear Unit"""
    return 1 * (x > 0)


def mse(expected: np.ndarray, actual: np.ndarray): 
    """Mean Squared Error"""
    return 0.5*np.linalg.norm(expected - actual)**2

def d_mse(expected: np.ndarray, actual: np.ndarray): 
    """Derivative of Mean Squared Error"""
    return expected - actual

@dataclass
class TrainingData:
    training: np.ndarray
    label: np.ndarray

class Network: 
    def __init__(self, shape: list[int], 
                training: np.ndarray, 
                labels: np.ndarray, 
                probe_training: np.ndarray=None, 
                probe_labels: np.ndarray=None, 
                suppress_print=False, 
                activation=sigmoid, 
                d_activation=d_sigmoid, 
                cost=mse, d_cost=d_mse): 
        """Initializes a new network."""
        if len(shape) < 2: 
            raise ValueError("Network needs at least 2 layers")

        if (shape[0] != training.shape[1]) or (shape[-1] != labels.shape[1]): 
            raise ValueError("Network shape does not match at inputs or outputs")

        conditions = [
            len(training.shape)  == 3, 
            len(labels.shape)  == 3,
            training.shape[2] == 1,
            labels.shape[2] == 1,
            training.shape[0] == labels.shape[0]
            ]
        if not all(conditions): 
            raise ValueError(
                "Input data and expected outputs should be 3 dimensional ndarrays \
                with the shape: (number of total examples, individual data size, 1)"
            )

        self.data = [TrainingData(t, l) for t, l in zip(training, labels)]
        if probe_training is None or probe_labels is None: 
            self.probe = None
        else:
            self.probe = [TrainingData(t, l) for t, l in zip(probe_training, probe_labels)]

        self.shape : list[int] = shape
        self.suppress_print : bool = suppress_print
        self.activation : Callable = activation
        self.d_activation : Callable = d_activation
        self.cost : Callable = cost
        self.d_cost : Callable = d_cost

        # creates empty lists which store weight 
        # matrices, and bias and activation vectors
        self.weights = [None] * len(self)
        self.biases = [None] * len(self)
        self.activations = [None] * (len(self) + 1)

        # setting random weights & biases [-0.5, 0.5)
        for i in range(len(self)): 
            self.weights[i] = np.random.rand(shape[i + 1], shape[i]) - 0.5
            self.biases[i] = np.random.rand(shape[i + 1], 1) - 0.5

    def __len__(self): 
        """Number of layers in the network."""
        return len(self.shape) - 1

    def _forwardprop(self, _input: np.ndarray): 
        """Propogrates a datapoint forward through the network to set activations."""
        # sets the input neurons to the input data
        self.activations[0] = _input
        # loops through each layer
        for l in range(len(self)): 
            # calculates the weighted sum and sets next activations based on activation function
            self.activations[l + 1] = self.activation(self.weights[l] @ self.activations[l] + self.biases[l])

    def _backprop(self, datapoint: TrainingData): 
        """Propogrates backwards through the network to find gradients."""
        # forward propogation
        self._forwardprop(datapoint.training)

        grad_biases = [None] * len(self)  
        grad_weights = [None] * len(self) 
        errors = [None] * len(self) 

        # runs backwards through the network
        for l in range(1, len(self) + 1): 
            # activation error is different on last layer
            a_err = self.d_cost(self.activations[-1], datapoint.label) if l == 1 else (self.weights[-l+1].T @ errors[-l+1])
            
            # error = dc/dz = dc/da * da/dz = a_err * dactiviation(z)
            errors[-l] = a_err * self.d_activation(self.weights[-l] @ self.activations[-l-1] + self.biases[-l])

            # computes the change in weights
            grad_weights[-l] = errors[-l] @ self.activations[-l-1].T
            grad_biases[-l] = errors[-l]

        return grad_weights, grad_biases

    def _train_batch(self, data: list[TrainingData], learning_rate): 
        """Trains the network on a certain batch of data."""
        # sets empty lists based on weight matrices and bias vectors
        sum_grad_weights =  [np.zeros(W.shape) for W in self.weights]
        sum_grad_biases =  [np.zeros(b.shape) for b in self.biases]

        # calculates the sum of the gradients for the batch
        for datapoint in data: 
            grad_weights, grad_biases = self._backprop(datapoint)
            sum_grad_weights = [mgw + gw for mgw, gw in zip(sum_grad_weights, grad_weights)]
            sum_grad_biases = [mgb + gb for mgb, gb in zip(sum_grad_biases, grad_biases)]

        # gradient descent
        k =  learning_rate/len(data)
        self.weights = [w - k*mgw for w, mgw in zip(self.weights, sum_grad_weights)]
        self.biases = [b - k*mgb for b, mgb in zip(self.biases, sum_grad_biases)]


    def train(self, epochs, batch_size, learning_rate):
        """Uses stochastic gradient descent to train the network"""
        if (batch_size > len(self.data)): 
            raise ValueError("Batch size too large")

        # trains over multiple epochs
        for epoch in range(1, epochs + 1):
            shuffled_data = random.sample(self.data, len(self.data))     

            # we splits the data into shuffled batches to get an
            # approxiate gradient to speed up the learning process
            for i in range(0, len(shuffled_data), batch_size): 
                self._train_batch(shuffled_data[i:i+batch_size], learning_rate)

            if not self.suppress_print: 
                print(self._summarize(epoch, epochs))



    def _summarize(self, epoch, total_epochs): 
        """Gives a string representation of an epoch"""
        if self.probe is None: 
            return f"Epoch {epoch}/{total_epochs}"
        return f"epoch {epoch}/{total_epochs} - accuracy: {self.accuracy(self.probe):.2f}% - loss: {self.loss():.2f}"

    def classify(self, datapoint): 
        """Classifies data by running it through the network."""
        self._forwardprop(datapoint)
        return self.activations[-1]

    def accuracy(self, data: list[TrainingData]):
        """The percentage of correct evaluations from a batch of data."""
        correct = [np.argmax(self.classify(d.training)) == np.argmax(d.label) for d in data]
        return  100*np.average(correct)


    def loss(self): 
        """The average cost from a batch of data."""
        return np.average([self.cost(self.classify(d.training), d.label) for d in self.data])
 
