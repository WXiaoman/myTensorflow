import tensorflow as tf

def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambda1)(var)
    )
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
layer_demension = [2, 10, 10, 10, 1]
n_layers = len(layer_demension)

cur_layer = x
# 当前节点个数
in_dimension = layer_demension[0]

# 生成5层的神经网络
for i in range(1, n_layers):
    out_dimension = layer_demension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_demension[i]

# 上面定义的时候把l2正则化的损失加到losses里了，下面把训练的损失加进去
mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses', mes_loss)
loss = tf.add_n(tf.get_collection('losses'))
print(loss)
