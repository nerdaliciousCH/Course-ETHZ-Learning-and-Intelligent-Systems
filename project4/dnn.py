import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics import accuracy_score
import sklearn.model_selection as sk
from tensorflow.examples.tutorials.mnist import input_data # TODO remove this
import random
from sklearn.decomposition import PCA

def main():

    class_count = 10;
    feature_count = 128;

    # Read input data from hd5 format
    train_labeled = pd.read_hdf("train_labeled.h5", "train")    # 9k rows
    train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")# 21k rows
    predict = pd.read_hdf("test.h5", "test")                    # 8k rows

    # Create matrices from input
    Y = train_labeled.iloc[0:,0]
    X = train_labeled.iloc[0:,1:]
    X_u = train_unlabeled.iloc[0:,1:]
    X_predict = predict.iloc[0:,0:]
    t_id = predict.index.values

    # Type conversion
    Y = np.array(Y.values).astype('int')
    X = np.array(X.values).astype('double')
    X_u = np.array(X_u.values).astype('double')
    X_predict = np.array(X_predict.values).astype('double')

    # To how many dimension do you wish to reduce?
    dim = 128

##    # Normalize
##    X -= np.mean(X, axis=0)
##    cov = np.dot(X.T, X)/ X.shape[0]
##
##    # Principal Component Analysis
##    U,S,V = np.linalg.svd(cov)
##    Xrot_reduced = np.dot(X,U[:,:dim])
##    X = Xrot_reduced
##
##    # Normalize
##    X_predict -= np.mean(X_predict, axis=0)
##    cov = np.dot(X_predict.T, X_predict)/ X_predict.shape[0]
##
##    # Principal Component Analysis
##    U,S,V = np.linalg.svd(cov)
##    Xrot_reduced = np.dot(X_predict,U[:,:dim])
##    X_predict = Xrot_reduced

    # Train-test split
    X_train, X_test, Y_train, Y_test = sk.train_test_split(X,Y,test_size=0.1, random_state=42)

    # Transform Y to one-hot-vector representation, i.e. 2 becomes [0,0,1,0,...,0], 0 becomes [1,0,...,] etc.
    Y_trainf = np.zeros((len(Y_train), class_count))
    Y_trainf[(range(len(Y_train)),Y_train)]=1
    Y_testf = np.zeros((len(Y_test), class_count))
    Y_testf[(range(len(Y_test)),Y_test)]=1

    logging.getLogger().setLevel(logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)


    # Define the network
    sess = tf.InteractiveSession()

    nodes_hidden_layer_1 = 256
    nodes_hidden_layer_2 = 2048
    nodes_hidden_layer_3 = 512
    nodes_hidden_layer_4 = 128

    # Define input and output variables
    x = tf.placeholder(tf.float32, shape=[None, dim])
    y_ = tf.placeholder(tf.float32, shape=[None, class_count])

    # Hidden Layer 1
    W_fc1 = weight_variable([dim, nodes_hidden_layer_1])
    b_fc1 = bias_variable([nodes_hidden_layer_1])
    y_fc1 = tf.matmul(x, W_fc1) + b_fc1

    # Hidden Layer 2
    W_fc2 = weight_variable([nodes_hidden_layer_1, nodes_hidden_layer_2])
    b_fc2 = bias_variable([nodes_hidden_layer_2])
    y_fc2 = tf.nn.tanh(tf.matmul(y_fc1, W_fc2) + b_fc2)

    # Hidden Layer 3
    keep_prob_1 = tf.placeholder(tf.float32)
    drop1 = tf.nn.dropout(y_fc2, keep_prob_1)
    W_fc3 = weight_variable([nodes_hidden_layer_2, nodes_hidden_layer_3])
    b_fc3 = bias_variable([nodes_hidden_layer_3])
    y_fc3 = tf.nn.relu(tf.matmul(drop1, W_fc3) + b_fc3)

    # Hidden Layer 4
    W_fc4 = weight_variable([nodes_hidden_layer_3, nodes_hidden_layer_4])
    b_fc4 = bias_variable([nodes_hidden_layer_4])
    y_fc4 = tf.nn.tanh(tf.matmul(y_fc3, W_fc4) + b_fc4)

    # Dropout and Output
    keep_prob_2 = tf.placeholder(tf.float32)
    drop2 = tf.nn.dropout(y_fc4, keep_prob_2)
    W_fc4 = weight_variable([nodes_hidden_layer_4, class_count])
    b_fc4 = bias_variable([class_count])
    y_conv = tf.matmul(drop2, W_fc4) + b_fc4
    
    sess.run(tf.global_variables_initializer())

    # Define Loss, Optimizer, Accuracy score
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate=0.0009).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())


    # Train the network
    print("Training...")
    batchsize = 100
    batches_in_training_set = Y_trainf.shape[0]/batchsize
    for j in range(2000):
      i = j % batches_in_training_set
      if(i == 0):
          idx = np.array(range(Y_trainf.shape[0]))
          random.shuffle(idx)
          Y_trainf = Y_trainf[idx]
          X_train = X_train[idx]
      start = i*batchsize
      end = min((i+1)*batchsize, Y_trainf.shape[0])
      batch = (X_train[start:end], Y_trainf[start:end])
      if j%batchsize == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob_1:1.0, keep_prob_2:1.0})
        print("Step %d, training accuracy %g"%(j, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob_1:0.5, keep_prob_2:0.5})

    print("TEST Accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: Y_testf, keep_prob_1:0.5, keep_prob_2:0.5}))

    
    print("Making predictions...")
    result = tf.argmax(y_conv.eval(feed_dict={x:X_predict, keep_prob_1:1.0, keep_prob_2:1.0}),1)
    predictions = result.eval()

    results_df = pd.DataFrame(data={'y':predictions})
    joined = pd.DataFrame({'Id':t_id}).join(results_df)

    # Save data in csv format for submission
    print("Writing predictions to predictions.csv")
    joined.to_csv("predictions.csv", index=False)

    

    


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
