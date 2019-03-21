import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_vaiables(shape, regularizer):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=1))
    # 如果有正则化函数，那么需要把这部分损失加入到名字是losses的集合里
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# 定义前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量，并前向传播
    with tf.variable_scope("layer1"):
        weights = get_weight_vaiables([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 类似地声明第二层
    with tf.variable_scope("layer2"):
        weights = get_weight_vaiables([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
