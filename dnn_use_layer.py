#通过添加层数来实现神经元，其中这个层数不是神经元的层数，而是两个神经元连接之间的交叉层数
#即，这里的层数不是代表可以看见的神经元，而是两个神经元之间的链接
#copy from youtube url: https://www.youtube.com/watch?v=S9wBMi2B4Ss&index=16&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8

import tensorflow as tf
import numpy as np

def add_layer(input, in_size, out_size, active_function=None):
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]), name='weight')
    bias = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(input, Weights) + bias  #matmul 不等于 tf.multiply
    if active_function is None:
        output = Wx_plus_b
    else:
        output = active_function(Wx_plus_b)
    return output

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
input_layer = add_layer(xs, 1, 10, active_function=tf.nn.relu)

prediction_layer = add_layer(input_layer, 10, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction_layer),axis=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i % 20 == 0:
        print('loss',sess.run(loss, feed_dict={xs:x_data,ys:y_data}))  #loss也需要feed值




