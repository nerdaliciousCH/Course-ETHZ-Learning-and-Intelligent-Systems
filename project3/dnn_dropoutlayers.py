import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics import accuracy_score
import sklearn.model_selection as sk
from tensorflow.examples.tutorials.mnist import input_data # TODO remove this

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # TODO remove this

    # read input data from hd5 format
    train = pd.read_hdf("train.h5", "train")
    predict = pd.read_hdf("test.h5", "test")

    # create matrices from input
    Y = train.iloc[0:,0]
    X = train.iloc[0:,1:]
    X_predict = predict.iloc[0:,0:]
    t_id = predict.index.values

    # type conversion
    Y = np.array(Y.values).astype('int')
    X = np.array(X.values).astype('double')
    X_predict = np.array(X_predict.values).astype('double')

    X_train, X_test, Y_train, Y_test = sk.train_test_split(X,Y,test_size=0.05, random_state=42)


    #logging.getLogger().setLevel(logging.INFO)
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 100])
    y_ = tf.placeholder(tf.float32, shape=[None,5])


    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    x_image = tf.reshape(x, [-1,10,10,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 5])
    b_fc2 = bias_variable([5])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    sess.run(tf.global_variables_initializer())

    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    # range(100) should be range(20000), run this at home
    for i in range(1000):
        # TODO THIS CALL SHOULD FRIGGIN WORK OR YOU ALL DIE
      batch = next_batch(X_train, Y_train, 0, i)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0}))

def next_batch(x, y, i, j):
    # TODO manidaag: write this function in a way that it does its magic in (TODO THIS CALL SHOULD FRIGGIN WORK OR YOU ALL DIE)
    batch = np.empty(shape=(j-i,2))
    batch[0] = x[i:j]
    batch[1] = y[i:j]
    return batch



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

if __name__ == '__main__':
    main()
