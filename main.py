from MNIST import labels, images
from Network import Network

if __name__ == '__main__': 

    training_data = images[:50_000]
    training_truths = labels[:50_000]

    probing_data = images[50_000:]
    probing_truths = labels[50_000:]

    nn = Network([784, 16, 16, 10], data=training_data, truth=training_truths)

    nn.train(epochs=100, batch_size=30, learning_rate=0.5, probe_data=probing_data, probe_truth=probing_truths)
