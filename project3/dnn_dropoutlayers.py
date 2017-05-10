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


    dim = 40
    

    X -= np.mean(X, axis=0)
    cov = np.dot(X.T, X)/ X.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot_reduced = np.dot(X,U[:,:dim])
    X = Xrot_reduced

    X_predict -= np.mean(X_predict, axis=0)
    cov = np.dot(X_predict.T, X_predict)/ X_predict.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot_reduced = np.dot(X_predict,U[:,:dim])
    X_predict = Xrot_reduced


    

    X_train, X_test, Y_train, Y_test = sk.train_test_split(X,Y,test_size=0.05, random_state=42)
    # TODO One Hot vector representation
    Y_trainf = np.zeros((len(Y_train), 5))
    Y_trainf[(range(len(Y_train)),Y_train)]=1
    Y_testf = np.zeros((len(Y_test), 5))
    Y_testf[(range(len(Y_test)),Y_test)]=1

    #logging.getLogger().setLevel(logging.INFO)
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, dim])
    y_ = tf.placeholder(tf.float32, shape=[None,5])

    W_fc1 = weight_variable([dim, 64])
    b_fc1 = bias_variable([64])
    y_fc1 = tf.matmul(x, W_fc1) + b_fc1

    W_fc2 = weight_variable([64, 128])
    b_fc2 = bias_variable([128])
    y_fc2 = tf.nn.relu(tf.matmul(y_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([128, 256])
    b_fc3 = bias_variable([256])
    y_fc3 = tf.nn.relu(tf.matmul(y_fc2, W_fc3) + b_fc3)

    W_fc4 = weight_variable([256, 32])
    b_fc4 = bias_variable([32])
    y_fc4 = tf.matmul(y_fc3, W_fc4) + b_fc4

    keep_prob = tf.placeholder(tf.float32)
    drop2 = tf.nn.dropout(y_fc4, keep_prob)
    W_fc4 = weight_variable([32, 5])
    b_fc4 = bias_variable([5])
    y_conv = tf.matmul(drop2, W_fc4) + b_fc4



    sess.run(tf.global_variables_initializer())


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    #cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # step = tf.Variable(1, trainable=False)
    # rate = tf.train.exponential_decay(1, step, 1, 0.9999)
    train_step = tf.train.AdamOptimizer(learning_rate=0.0009).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
<<<<<<< Updated upstream
    # TODO change this to real max entries of train set
    batchsize = 100
    
    for j in range(8000):
      i = j % 431
      if(j % 431 == 0):
          idx = np.array(range(Y_trainf.shape[0]))
          random.shuffle(idx)
          Y_trainf = Y_trainf[idx]
          X_train = X_train[idx]
      start = i*batchsize
      end = min((i+1)*batchsize, Y_trainf.shape[0])
      batch = (X_train[start:end], Y_trainf[start:end])
<<<<<<< Updated upstream
||||||| ancestor
    # range(100) should be range(20000), run this at home
    for i in range(1000):
        # TODO THIS CALL SHOULD FRIGGIN WORK OR YOU ALL DIE
      batch = next_batch(X_train, Y_train, 0, i)
=======
    # range(100) should be range(20000), run this at home
    for i in range(1000):
        # TODO THIS CALL SHOULD FRIGGIN WORK OR YOU ALL DIE
      batch = next_batch(X_train, Y_train, 0, 100)
>>>>>>> Stashed changes
      if i%100 == 0:
=======
      if j%100 == 0:
>>>>>>> Stashed changes
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(j, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.7})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: Y_testf, keep_prob:0.7}))

    
    print("Making predictions...")
    result = tf.argmax(y_conv.eval(feed_dict={x:X_predict, keep_prob:1.0}),1)
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
