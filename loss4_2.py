import tensorflow as tf

from numpy.random import RandomState

batch_size = 8
# 输入两个节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般输出一个节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
xx = rdm.rand(dataset_size, 2)
# x1和x2加一个随机数，随机数是均值为0的小量，范围是-0.05到0.05
yy = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in xx]

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: xx[start:end], y_: yy[start:end]})
        print(sess.run(w1))
