import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE= 28
NUM_CHANNELS  = 1
NUM_LABELS = 1

CONV1_DEEP = 32
CONV1_SIZE =  5

CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 5
# LAYER1_NODE = 500

# 定义前向传播过程
def inference(input_tensor, train, regularizer):
    # 声明第一层神经网络的变量，并前向传播
    with tf.variable_scope("layer1-conv1"):
        # 前两个参数是过滤器的尺寸，第三个是当前的深度，第四个是过滤器的深度
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        # print(pool_shape)
        #  计算被拉长成向量后向量的长度。pool_shape[0]是一个batch_size的长度
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # print(pool_shape[0])   为空
        # print(pool_shape[1])
        # print(pool_shape[2])   7
        # print(nodes)    3136
        # print(pool_shape[3])   64
        # reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
        reshaped = tf.reshape(pool2, [100, nodes])


    with tf.variable_scope("layer5-fc1"):
        # 输入的是拉直后的一个向量，长度为3136。输入是长度为512的向量
        fc1_weight = tf.get_variable("wieght", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weight))
        fc1_biases = tf.get_variable("biases", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_biases)
        # dropout可以防止过拟合，在输出的时候随机的把一些输出改为0
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weight = tf.get_variable("wieght", [FC_SIZE, NUM_LABELS],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weight))
        fc2_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc1, fc2_weight) + fc2_biases
        # dropout可以防止过拟合，在输出的时候随机的把一些输出改为0
    return logits
