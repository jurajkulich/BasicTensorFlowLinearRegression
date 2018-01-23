import tensorflow as tf
import numpy
import matplotlib.pyplot as plt


session = tf.Session()

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = numpy.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + numpy.random.normal(0.0, 0.3)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
session.run(init)

for step in range(50):
    session.run(train)
    print(step, session.run(W), session.run(b))
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, session.run(W) * x_data + session.run(b))
    plt.legend()
    plt.show()

