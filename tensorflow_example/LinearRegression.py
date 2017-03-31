#tensorflow linear regression example
#

import tensorflow as tf

# X, Y
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#hypothesis
hypothesis = X * W + b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

#Launch
with tf.Session() as sess:

    #initial
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                             feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
