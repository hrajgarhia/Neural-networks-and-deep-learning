#print(tensorflow.__version__)

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
# inside the with block, the session is set as the default session. Calling x.initializer.run() is equivalent to
# calling tf.get_default_session().run(x.initializer)

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
    result = f.eval()


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


# compare the above calculation of theta by doing it with pure numpy

X=housing_data_plus_bias
y=housing.target.reshape(-1,1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# compare the above calculation of theta by doing it with sklearn

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(housing.data,housing.target.reshape(-1,1))
print(np.r_[lin_reg.intercept_.reshape(-1,1),lin_reg.coef_.T])


# Linear regression gradient descent solution using tensorflow


# Gradient descent requires scaling the feature vectors first.

from sklearn.preprocessing import StandardScaler
scalar =  StandardScaler()
scaled_housing_data =  scalar.fit_transform(housing.data)
scaled_housing_data_plus_bias= np.c_[np.ones((m,1)),scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed =42),name='theta')
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

# Gradient descent using autodiff

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients =  tf.gradients(mse,[theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)


# practicing autodiff

def my_func(a,b):
    z=0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z

my_func(0.2,0.3)

# let's compute the function at a = 0.2 and b = 0.3 and the partial derivative
# of that point with regards to a and b respectively

reset_graph()
a =tf.Variable(0.2,name='a')
b = tf.Variable(0.3,name='b')
z= tf.constant(0.0,name='z0')

for i in range(100):
    z = a * tf.cos(z + i) + z * tf.sin(b - i)

grads = tf.gradients(z,[a,b])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(z.eval())
    print(sess.run(grads))

# Using an GradientDescentOptimizer


reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer =tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)


# Using an MomentumOptimizer


reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)

# Placeholders

# Placeholders are special because they don't actually perform any computation,
# they just ouput the data we tell them to output at runtime


# Placeholder practice

A = tf.placeholder(tf.float32, shape =(None,3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A:[[4,5,6],[7,8,9]]})
print(B_val_1)
print(B_val_2)

# Mini - Batch Gradient Descent

n_epochs = 10
learning_rate = 0.01
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch,batch_index,batch_size):
    np.random.seed(epoch*n_batches+batch_index)
    indices = np.random.randint(m,size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1,1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch,batch_index,batch_size)
            sess.run(training_op, feed_dict = {X: X_batch,y:y_batch})
    best_theta = theta.eval()
