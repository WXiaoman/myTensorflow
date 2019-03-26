import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference6
import numpy as np

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99

# MODEL_PATH = "./mnistTrain"
# MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入placeholder
    # x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    x = tf.placeholder(tf.float32,
                       [None, mnist_inference6.IMAGE_SIZE,
                        mnist_inference6.IMAGE_SIZE, mnist_inference6.NUM_CHANNELS],
                       name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference6.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 用mnist_inference定义的前向传播
    y = mnist_inference6.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作和训练过程
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True
    )
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 训练时，每过一次数据都既需要通过反向传播更新参数，又需要更新参数的滑动平均值。
    # 下面等于train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 训练过程不测试在验证集上的结果，验证和测试都会有独立的过程
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference6.IMAGE_SIZE,
                                          mnist_inference6.IMAGE_SIZE,
                                          mnist_inference6.NUM_CHANNELS
                                          ))
            # _, loss_value, step = sess.run([train_op, loss, global_step],
            #                                feed_dict={x: xs, y_: ys})
            # rexs, rexy = sess.run([reshaped_xs, ys])
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                # 这里只输出损失函数大小
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # saver.save(
                #     sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step
                # )

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()
