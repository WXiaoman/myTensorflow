import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 测试时不关注正则化损失的值，所以用于计算正则化损失的函数设置为none
        y = mnist_inference.inference(x, None)
        # 用前向传播的结果计算正确率，
        # 如果需要预测未知例子，用tf.argmax(y, 1)就可以得到结果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 用变量重命名的方法加载模型，这样在前向传播过程就不需要再调用求滑动平均的函数了
        # 这样就可以完全用inference中定义的前向传播过程
        variable_average = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每隔10s调用一次计算正确率的过程来检测训练过程中的正确率变化
        while True:
            with tf.Session() as sess:
                # get_checkpoint_state会通过checkpoint文件自动找到目录中最新的模型文件名
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), acc is %g." % (global_step, accuracy_score))
                else:
                    print("no checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
