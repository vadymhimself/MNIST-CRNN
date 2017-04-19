import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras import metrics
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import GRU
from keras.utils.np_utils import to_categorical

# for reproducibility
np.random.seed(2017)

SEQ_LENGTH = 8
EPOCHS = 30
EXAMPLES_PER_EPOCH = 500
BATCH_SIZE = 32

# the data, shuffled and split between train and test sets
(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()

# basic image processing
# convert images to float
X_train_raw = X_train_raw.astype('float32')
X_test_raw = X_test_raw.astype('float32')
X_train_raw /= 255
X_test_raw /= 255

# encode output
y_train_raw = to_categorical(y_train_temp, 10)
y_test_raw = to_categorical(y_test_temp, 10)

train_size, height, width = X_train_raw.shape
depth = 1

# define our time-distributed setup
model = Sequential()
model.add(TimeDistributed(Convolution2D(8, 4, strides=4, padding='valid', activation='relu'),
                          input_shape=(SEQ_LENGTH, height, width, depth)))
model.add(TimeDistributed(Convolution2D(16, 3, strides=3, padding='valid', activation='relu')))
model.add(TimeDistributed(Flatten()))
model.add(GRU(50, return_sequences=True, dropout=.3))
model.add(TimeDistributed(Dense(10, activation='softmax')))

model.compile(loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy], optimizer='rmsprop')


def generate_rand_sequences(max_seq_length, count):
    x = np.zeros((count, max_seq_length, height, width, depth))
    y = np.zeros((count, max_seq_length, 10))

    for i in range(0, count):
        # decide how many MNIST images to put in that tensor
        curr_seq_length = int(np.ceil(np.random.rand() * SEQ_LENGTH))
        # sample that many images
        indices = np.random.choice(X_train_raw.shape[0], size=curr_seq_length)

        # initialize a training example
        example_x = np.zeros((max_seq_length, height, width, depth))
        example_x[0:curr_seq_length, :, :, 0] = X_train_raw[indices]
        x[i, :, :, :, :] = example_x

        example_y = np.zeros((max_seq_length, 10))
        example_y[0:curr_seq_length, :] = y_train_raw[indices]
        y[i, :, :] = example_y

        np.append(x, example_x)
        np.append(y, example_y)

    return x, y


def show_images(imgs, labels):
    fig = plt.figure()
    number_of_files = imgs.shape[0]
    for i in range(number_of_files):
        fig.add_subplot(1, number_of_files, i + 1).set_title(labels[i])
        plt.imshow(imgs[i], cmap='Greys_r')
        plt.axis('off')
    plt.show()


# run epochs of sampling data then training
for ep in range(0, EPOCHS):
    X_train, Y_train = generate_rand_sequences(SEQ_LENGTH, EXAMPLES_PER_EPOCH)

    if ep == 0:
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", Y_train.shape)

    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=1, verbose=1, validation_split=.1)
    model.save('model_epoch-{}.HDF5'.format(ep))

# manual test
test_x, test_y = generate_rand_sequences(SEQ_LENGTH, 1)
# print test_x.shape
eight_images = np.reshape(test_x[0], (-1, 28, 28))
prediction = [np.argmax(x) for x in model.predict(test_x, batch_size=1)[0]]
show_images(eight_images, prediction)
plt.show()
