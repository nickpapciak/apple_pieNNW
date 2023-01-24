from MNIST import labels, images
from Network import Network

if __name__ == '__main__': 
    nn = Network([784, 40, 40, 10], training= images[:50_000], labels=labels[:50_000], probe_training=images[50_000:], probe_labels=labels[50_000:])
    nn.train(epochs=10, batch_size=15, learning_rate=0.1)

# check suppress
