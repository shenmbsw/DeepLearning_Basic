from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def logistic_regression(x_):
    # create the actual model
    scope_args = {'initializer': tf.random_normal_initializer(stddev=1e-4)}
    with tf.variable_scope("weights", **scope_args):
        W = tf.get_variable('W', shape=[784, 10])
        b = tf.get_variable('b', shape=[10])
        y_logits = tf.matmul(x_, W) + b
    return y_logits

def test_classification(model_function, learning_rate=0.1):
    # import data
    mnist = read_data_sets('./datasets/mnist/', one_hot=True)

    with tf.Graph().as_default() as g:
        # where are you going to allocate memory and perform computations
        with tf.device("/cpu:0"):
            
            # define model "input placeholders", i.e. variables that are
            # going to be substituted with input data on train/test time
            x_ = tf.placeholder(tf.float32, [None, 784])
            y_ = tf.placeholder(tf.float32, [None, 10])
            y_logits = model_function(x_)
            
            # naive implementation of loss:
            # > losses = y_ * tf.log(tf.nn.softmax(y_logits))
            # > tf.reduce_mean(-tf.reduce_sum(losses, 1))
            # can be numerically unstable.
            #
            # so here we use tf.nn.softmax_cross_entropy_with_logits on the raw
            # outputs of 'y', and then average across the batch.
            
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits)
            cross_entropy_loss = tf.reduce_mean(losses)
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
            correct_prediction = tf.equal(y_pred, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with g.as_default(), tf.Session() as sess:
        # that is how we "execute" statements 
        # (return None, e.g. init() or train_op())
        # or compute parts of graph defined above (loss, output, etc.)
        # given certain input (x_, y_)
        sess.run(tf.initialize_all_variables())
        
        # train
        for iter_i in range(20001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})
            
            # test trained model
            if iter_i % 2000 == 0:
                tf_feed_dict = {x_: mnist.test.images, y_: mnist.test.labels}
                acc_value = sess.run(accuracy, feed_dict=tf_feed_dict)
                print('iteration %d\t accuracy: %.3f'%(iter_i, acc_value))
                
test_classification(logistic_regression, learning_rate=0.1)