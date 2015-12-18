#!/usr/bin/env python

import tensorflow as tf
from toy_datasets import *
from neural_network import *
from utils import *
from activation_functions import *
import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# get data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start the session
sess = tf.InteractiveSession()

# placeholder for data and targets
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

## softmax classifier ##
'''
# params
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# init variables
sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
'''

## conv net ##

# reshape x
x_image = tf.reshape(x, [-1, 28, 28, 1])


# make conv layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Evaluate
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in xrange(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step {0}, training accuracy {1}".format(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

print "test accuracy {}".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))





#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#batch = mnist.train.next_batch(50)

#for i in xrange(1000):
#    batch = mnist.train.next_batch(50)
#    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# generating test data
#X, Y = generate_2_class_moon_data()
#X2, Y2 = generate_2_class_moon_data()
#X, Y = generate_3_class_spiral_data(nb_classes=3, theta=0.5, plot=False)
#X2, Y2 = generate_3_class_spiral_data(nb_classes=3, theta=0.5, plot=False)
#X, y = load_iris_dataset()

# digit dataset
#train, test = load_digit_dataset(500, 0.1)
#X = np.array([x[0] for x in train])
#Y = [y[1] for y in train]
#X2 = np.array([x[0] for x in test])
#Y2 = [y[1] for y in test]


# running demo models
#m = build_model2(X, y, 3, 50)
#m = build_model1(train_data=X, labels=y, nb_classes=10, nn_hdim=50, print_loss=True)
#plot_decision_boundary(lambda x: predict(m, x), X, y)

# testing library

#net = NeuralNetwork([2, 10, 2], hyperbolic_tangent)

#net = NeuralNetwork(input_dim=X.shape[1],
#                    nb_classes=len(set(Y)),
#                    hidden_dims=[100],
#                    activation_function=hyperbolic_tangent)

#net.mini_batch_sgd(training_data=X,
#                   labels=Y,
#                   epochs=5000,
#                   batch_size=10,
#                   epsilon=0.001,
#                   lbda=0.001)


#net.fit(X, Y, epochs=5000, epsilon=0.001, lbda=0.001, print_loss=True)
#t = net.evaluate(X2, Y2)
#print net.predict_old(X2)[1:10]
#print net.predict(X2)[1:10]
#print net.predict_old(X2)[1:10] == net.predict(X2)[1:10]
#plot_decision_boundary(lambda x: np.argmax(net.predict(x), axis=1),
#                       X, Y)

