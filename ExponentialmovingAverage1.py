import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)
# step模拟神经网络中迭代的轮数， 可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_average_op = ema.apply([v1])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([v1, ema.average(v1)]))

    # 更新v1成5
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 1000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
