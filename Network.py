import numpy as np 
import random 

# TODO: use python typing to do actual types 
# TODO: implement momentum
# TODO: make output look nicer like keras https://www.tensorflow.org/tutorials/keras/classification

# assorted functions
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def dsigmoid(x): 
    return sigmoid(x)*(1-sigmoid(x))

def mean_squared_error(expected, actual): 
    return 0.5*np.linalg.norm(expected - actual)**2

def dmean_squared_error(expected, actual): 
    return expected - actual

class Network: 
    def __init__(
        self, shape, data, truth, 
        activation_function=sigmoid, 
        dactivation_function=dsigmoid, 
        cost_function=mean_squared_error, 
        dcost_function=dmean_squared_error
        ): 
        """Initializes a new network."""
        if len(shape) < 2: 
            raise ValueError("Network needs at least 2 layers")

        if (shape[0] != data.shape[1]) or (shape[-1] != truth.shape[1]): 
            raise ValueError("Network shape does not match at inputs or outputs")

        if any((
            len(data.shape) != 3, 
            len(truth.shape) != 3, 
            data.shape[2] != 1, 
            truth.shape[2] != 1, 
            data.shape[0] != truth.shape[0]
            )): 
            raise ValueError(
                "Input data and expected outputs should be 3 dimensional ndarrays \
                with the shape: (number of total examples, individual data size, 1)"
            )

        self.shape = shape
        self.data = data
        self.truth = truth
        self.activation_function = activation_function
        self.dactivation_function = dactivation_function
        self.cost_function = cost_function
        self.dcost_function = dcost_function

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

    def _forwardprop(self, data): 
        """Propogrates data forward through the network to set activations."""
        # sets the input neurons to the input data
        self.activations[0] = data
        # loops through each layer
        for l in range(len(self)): 
            # calculates the weighted sum and sets next activations based on activation function
            self.activations[l + 1] = self.activation_function(self.weights[l] @ self.activations[l] + self.biases[l])

    def _backprop(self, data, truth): 
        """Propogrates backwards through the network to find gradients."""
        # forward propogation
        self._forwardprop(data)

        grad_biases = [None] * len(self)  
        grad_weights = [None] * len(self) 
        errors = [None] * len(self) 

        # runs backwards through the network
        for l in range(1, len(self) + 1): 
            # activation error is different on last layer
            a_err = self.dcost_function(self.activations[-1], truth) if l == 1 else (self.weights[-l+1].T @ errors[-l+1])
            
            # error = dc/dz = dc/da * da/dz = a_err * dactiviation(z)
            errors[-l] = a_err * self.dactivation_function(self.weights[-l] @ self.activations[-l-1] + self.biases[-l])

            # computes the change in weights
            grad_weights[-l] = errors[-l] @ self.activations[-l-1].T
            grad_biases[-l] = errors[-l]

        return grad_weights, grad_biases

    def _batch_train(self, batch_data, batch_truth, learning_rate): 
        """Trains the network on a certain batch of data."""
        # sets empty lists based on weight matrices and bias vectors
        sum_grad_weights =  [np.zeros(W.shape) for W in self.weights]
        sum_grad_biases =  [np.zeros(b.shape) for b in self.biases]

        # calculates the sum of the gradients for the batch
        for data, truth in zip(batch_data, batch_truth): 
            grad_weights, grad_biases = self._backprop(data, truth)
            sum_grad_weights = [mgw + gw for mgw, gw in zip(sum_grad_weights, grad_weights)]
            sum_grad_biases = [mgb + gb for mgb, gb in zip(sum_grad_biases, grad_biases)]

        # gradient descent
        k =  learning_rate/len(batch_data)
        self.weights = [w - k*mgw for w, mgw in zip(self.weights, sum_grad_weights)]
        self.biases = [b - k*mgb for b, mgb in zip(self.biases, sum_grad_biases)]

    def train(self, epochs, batch_size, learning_rate, probe_data=None, probe_truth=None):
        """Uses stochastic gradient descent to train the network"""
        if (batch_size > len(self.data)): 
            raise ValueError("Batch size too large")

        for e in range(1, epochs + 1):
            # splits the data into random smaller batches and then trains
            # on these smaller random samples to approximate the gradient
            shuffled_data = list(zip(self.data, self.truth))
            random.shuffle(shuffled_data)
            batch_data, batch_truth = zip(*shuffled_data) # undoes the zip
            
            # chunks the data and the truth into batches of size `batch size`
            for i in range(0, len(batch_data), batch_size): 
                self._batch_train(batch_data[i:i+batch_size], batch_truth[i:i+batch_size], learning_rate)
            
            # prints the epoch
            print(f"Epoch {e}") if probe_data is None and probe_truth is None else \
            print(f"Epoch {e} ({100*self.evaluate(probe_data, probe_truth)/len(probe_data):.2f}%) \
                    Cost ({self.cost(batch_data, batch_truth):.2f})"
            )

    def classify(self, data): 
        """Classifies data by running it through the network."""
        self._forwardprop(data)
        return self.activations[-1]

    def evaluate(self, batch_data, batch_truth):
        """The number of correct evaluations from a batch of data."""
        return sum([np.argmax(self.classify(d)) == np.argmax(t) for d, t in zip(batch_data, batch_truth)])

    def cost(self, data, expected): 
        """The current network cost from a batch of data."""
        return self.cost_function(self.classify(data), expected)
