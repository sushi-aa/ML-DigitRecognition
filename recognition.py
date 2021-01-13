
import numpy
import os
import gzip
import matplotlib
import matplotlib.pyplot as pt
import lasagne
import theano
import theano.tensor as tens
import urllib.request

#https://github.com/PacktPublishing/From-0-to-1-Machine-Learning-NLP-Python-Cut-to-the-Chase/blob/master/Section%2015/DigitRecognition-Python3.py
#https://www.youtube.com/watch?v=lbFEZAXzk0g


def load_dataset():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading ", filename)
        urllib.request.urlretrieve(source+filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28)
            return data/numpy.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
        return data

    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()

matplotlib.use('TkAgg')
pt.imshow(x_train[3][0])
pt.show()


def buildNN(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=10, nonlinearity = lasagne.nonlinearities.softmax)

    return l_out

input_var = tens.tensor4('inputs')
target_var = tens.ivector('targets')

network = buildNN(input_var)

prediction = lasagne.layers.get_output(network)
print(prediction)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

for step in range(10):
    train_err = train_fn(x_train, y_train)

test_prediction = lasagne.layers.get_output(network)
val_fn = theano.function([input_var], test_prediction)
val_fn([x_test[0]])

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_acc = tens.mean(tens.eq(tens.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
acc_fn = theano.function([input_var, target_var], test_acc)

acc_fn(x_test, y_test)
