print(tensorflow.__version__)

import tensorflow as tf


x=tf.Variable(3,name='x')
y=tf.Variable(4,name='y')
f=x*x*y + y + 2

# let draw our first tensorflow computation graph


sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result =  sess.run(f)
print(result)
sess.close()

# Writing sess.run() multiple times can be cumbersome. To avoid that we can use "with" which will automatically close
# the session when it runs and evaluates the results

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

# Instead of manually running the initializer several times, we can use global_variables_initializer().
# The function does not initialises the variables immediately but creates a node in the graph which will initialise
# the variables when it is run

init=tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()


#Inside Jupyter or within a Python shell you may prefer to create an InteractiveSession. The only
#difference from a regular Session is that when an InteractiveSession is created it automatically sets
#itself as the default session, so you don’t need a with block (but you do need to close the session
#manually when you are done with it)

sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()


#When you evaluate a node, TensorFlow automatically determines the set of nodes that it depends on and it
#evaluates these nodes first. For example, consider the following code:
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15

#The above code will evaluate w and x twice

#All node values are dropped between graph runs, except variable values, which are maintained by the
#session across graph runs. A variable starts its life when its initializer is run, and it ends when the session
#is closed.

#If you want to evaluate y and z efficiently, without evaluating w and x twice as in the previous code, you
#must ask TensorFlow to evaluate both y and z in just one graph run, as shown in the following code:


with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15


# In single-process TensorFlow, multiple sessions do not share any state, even if they reuse the same graph (each
# session would have its own copy of every variable). In distributed TensorFlow, variable state is stored on the
# servers, not in the sessions, so multiple sessions can share the same variables.

# Linear regression closed form solution using tensorflow
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()


# Linear regression gradient descent solution using tensorflow

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
y_pred = tf.matmul(X,theta,name='predictions')
error = y_pred-y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
             print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

