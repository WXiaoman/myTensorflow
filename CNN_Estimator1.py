# 用Estimator构建CNN，识别mnist数据集中的数字
# 网页链接https://www.tensorflow.org/tutorials/estimators/cnn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # [batch_size, image_height, image_width, channels]data_format
    # chanel通道是颜色，单色图为1，彩色图为3红绿蓝
    # channels_last（默认）或channels_first之一。
    # channels_last对应于形状为(batch, ..., channels)的输入
    # channels_first对应于形状为(batch, channels, ...)的输入。
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #卷积层
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        # 指定输出张量与输入张量具有相同的高度和宽度值，所以设置same
        # 表示TensorFlow向输入张量的边缘添加0值，使高度和宽度均保持为28
        # 没有填充的话，在28x28张量上进行5x5卷积运算将生成一个24x24张量，
        # 因为在28x28网格中，可以从24x24个位置提取出一个5x5图块
        padding="same",
        activation=tf.nn.relu
    )

    # 池化层
    # 生成的输出张量的形状为：2x2过滤器将高度和宽度各减少50％。[batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2
    )

    #卷积层
    # 输出[batch_size，14，14，64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    #池化层
    # 输出[batch_size，7，7，64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2
    )

    # 扁平化pool2输出的特征图，使其变形为只有两个维度[batch_size, features]
    # -1表示动态计算，7*7是pool2输出的高度和宽度，64是pool2输出的通道数
    # features=7*7*64=3136
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    #密集层
    # units是密集层的神经元数量
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # 丢弃正则化
    # rate是丢弃率，这里是40%的元素会在训练中随机被丢弃
    # 只有training是true才会执行丢弃操纵。检查传到模型的mode是否为train模式
    # 输出张量为[batch_size, 1024]
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # 对数层，返回预测的原始值。
    logits = tf.layers.dense(inputs=dropout, units=10)
    # 最终输出张量[batch_size, 10]

    predictions = {
        # axis表示沿着索引为1的维度查找最大值，该维度对应于预测结果，已经知道对数张量的形状为[batch_size, 10]
        "classes": tf.argmax(input=logits, axis=1),
        # 使用name参数明确将该操作命名为softmax_tensor，以便稍后引用它
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 对于像MNIST这样的多类别分类问题，将通常交叉熵用作损失指标
    # labels包含预测结果0-9，logits包含最后一层的线性输出
    # sparse_softmax_cross_entropy高效计算交叉熵，也就是负对数似然率
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 在评估模式下定义字典搜索。添加了准确率指标。
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

# 训练和评估CNN MNIST分类器
def main(unused_argv):
    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # 在train_data和train_labels中将训练特征数据（55000张手写数字图像的原始像素值）
    # 和训练标签（每张图像在0到9之间的对应值）分别存储为Numpy数组
    train_data = mnist.train.images # 返回np.arrays
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # 将评估特征数据（ 10000张图像）和评估标签分别存储在eval_data和eval_labels中。
    eval_data = mnist.test.images # 返回np.arrays
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # 创建Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./mnist_convnet_model"
    )
    # 设置日志来记录过程softmax层的概率值
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
        # 每完成50个训练周期就记录一次概率
    )

    # 训练模型
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},   # 作为字典传递
        y=train_labels,
        batch_size=100,   # 每一步训练100个小批次样本
        num_epochs=None,  # 表示会一直训练，大啊都指定的训练步数
        shuffle=True   # 训练期间触发logging_hook
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook]
    )

    # 评估模型
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False   # 表示按顺序遍历数据
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
