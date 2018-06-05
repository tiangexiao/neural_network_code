#cnn网络当中卷积层的参数w是卷积核，在图像中卷积为4维的，为filter_height, filter_width, in_channels, out_channels
#全连接层的参数和普通的dnn网络中的w是一样的，in_shape,out_shape的形状
#youtube url: https://www.youtube.com/watch?v=pjjH2dGGwwY&index=28&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np


print(mnist.train.labels.shape)
print(mnist.train.images.shape)


def add_layer(input, in_size, out_size, active_function=None):
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]), name='weight')
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + bias  # matmul 不等于 tf.multiply
    if active_function is None:
        output = Wx_plus_b
    else:
        output = active_function(Wx_plus_b)
    return output

def compute_accuracy(v_x, v_y):
    y_pred = sess.run(prediction, feed_dict={xs:v_x,keep_prob:1})  #非常神奇，这里不需要定义prediction还有sess竟然可以运行
    correct_predict = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    result = sess.run(accuracy)
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32, shape=[None, 28 * 28])
ys = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

#conv1 layer
w1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
conv1 = tf.nn.relu(conv_layer(x_image, w1) + b1)  # 28,28,1 ->> 28, 28 ,32
pool1 = max_pool_2x2(conv1)                       #28,28,32 ->>14, 14, 32

#conv2 layer
w2 = weight_variable([4,4,32,64])
b2 = bias_variable([64])
conv2 = tf.nn.relu(conv_layer(pool1, w2)+b2)
pool2 = max_pool_2x2(conv2)       #7,7,64



#fc1 layer
reshape_pool2 = tf.reshape(pool2, [-1,7*7*64])

w3 = weight_variable([7*7*64, 1024])
b3 = weight_variable([1024])
conv3 = tf.nn.relu(tf.matmul(reshape_pool2, w3) + b3)
dropped_conv3 = tf.nn.dropout(conv3, keep_prob)


#fc2 layer
w4 = weight_variable([1024, 10])
b4 = weight_variable([10])
prediction = tf.nn.softmax(tf.matmul(dropped_conv3, w4) + b4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.global_variables_initializer()
writer = tf.summary.FileWriter('logs')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
sess.run(init)
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
