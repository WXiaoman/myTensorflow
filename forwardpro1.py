# import tensorflow as tf
#
# w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
#
# # x = tf.constant([[0.7, 0.9]])
# x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
#
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
#
# with tf.Session() as sess:
#     # sess.run(w1.initializer)
#     # sess.run(w2.initializer)
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
#

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x_input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y_input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数和反向传播算法
y = tf.sigmoid(y)
# 交叉熵
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 随机数生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
# 生成dataset_set行，2列的数据
xx = rdm.rand(dataset_size, 2)
# 正样本
yy = [[int(xx1 + xx2 < 1)] for (xx1, xx2) in xx]

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 5000
    # print(w1)
    # print(w2)
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: xx[start:end], y_: yy[start:end]})
        if i % 1000 == 0:
          total_cross_entropy = sess.run(cross_entropy, feed_dict={x: xx, y_: yy})
          print("after %d training step(s), total cross entropy is %g" % (i, total_cross_entropy))

    # print(w1)
    # print(w2)
    print(sess.run(w1))
    print(sess.run(w2))
