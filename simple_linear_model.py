"""
定义两个可以更新的Variable：W和b
定义损失函数loss
使用迭代器更新loss
"""
import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='weights')
biases = tf.Variable(tf.zeros([1]), name='biases')

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)



init = tf.global_variables_initializer()

#查看可以进行训练的参数
for variable in tf.trainable_variables():
    print(variable)

sess = tf.Session()

sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases), sess.run(loss))
        
