import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# 辅助函数，给定神经网络的输入和参数，计算前向传播的结果。
# 定义relu激活的全连接三层神经网络，通过加入隐藏层实现多层网络结构
# 在这个函数中支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，就使用当前的参数
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数计算得出变量的滑动平均值，再计算神经网络的前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")
    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 计算当前参数下神经网络的前向传播结果，因为这里给出的用于计算的滑动平均的类为none, 所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义训练轮次，这里不使用滑动平均值，所以trainable=False
    global_step = tf.Variable(0, trainable=False)
    # 定义滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络的参数上使用滑动平均，其他辅助变量（如global_step)不使用tf.trainable_variables返回的就是
    # 图上集合GraphKeys.TRAINABLE_VARIABLES中的元素，这个集合的元素就是所有没有指定trainable=False的元素
    variable_averages_op = variable_average.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_average, weights1, biases1, weights2, biases2)

    # 这个函数的第一个参数是不包括softmax层的前向传播结果，第二个是训练数据的正确答案。
    # 因为标准答案是一个长度为10的一维数组，该函数需要提供的是一个正确答案的数字，
    # 所以需要用tf.argmax函数来得到正确答案对应的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上的权重的正则化损失，而不计算bias
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY
    )
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 训练时，每过一次数据都既需要通过反向传播更新参数，又需要更新参数的滑动平均值。
    # 下面等于train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确
    correct_predictions = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 把bool型转换为实数型之后计算平均值，就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代训练
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("after %d training step(s), validation accuracy using  average model is %g" % (i, validate_acc))
            # 每一轮使用一个batch的数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 训练结束后，在测试数据集上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("test: after %d training step(s), validation accuracy using  average model is %g" % (i, test_acc))


def main(argv=None):
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    mnist = input_data.read_data_sets("./", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
