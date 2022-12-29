import numpy as np

# number of images to load from the MNIST dataset
NUM_IMGS = 60000
# size of each square image in the dataset
IMG_SIZE = 28

# gets the labels from MNIST dataset 
with open('data/MNIST/labels','rb') as f: 
    f.seek(8)
    humanized_labels = list(f.read())[:NUM_IMGS]

    # converts the list of values i.e. [5, 0, 4, 1, 9, 2 ... ] 
    # into a list of vectors with 1 at the corresponding index
    labels = np.zeros((NUM_IMGS,10, 1))
    for i, label in enumerate(humanized_labels): 
        labels[i][label] = 1


# gets the images from MNIST dataset 
with open('data/MNIST/imgs','rb') as f: 
    f.seek(16)
    images = list(f.read())[:NUM_IMGS*IMG_SIZE**2]
    images = np.reshape(images, (NUM_IMGS, IMG_SIZE**2, 1))
    images = images / 255

# for testing
# displays an image from an image vector
# def show_img(img):
#     from matplotlib import pyplot as plt
#     plt.imshow(np.reshape(img, (IMG_SIZE, IMG_SIZE)))
#     plt.show()

