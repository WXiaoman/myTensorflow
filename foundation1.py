# 基础
# 网页链接  https://www.tensorflow.org/tutorials/eager/custom_training

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# 定义模型


class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.B = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.B

# model = Model()
# print(model(3.0))
# assert model(3.0).numpy() == 15.0


def loss(predict_y, desire_y):
    return tf.reduce_mean(tf.square(predict_y-desire_y))

# 训练数据。用噪声合成数据


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
output = inputs * TRUE_W + TRUE_b + noise

# plt.scatter(input, output, c='b')
# plt.scatter(input, model(input), c='r')
# plt.show()


def train(model, inputs, output, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs),output)
    dW, db = t.gradient(current_loss, [model.W, model.B])
    model.W.assign_sub(learning_rate * dW)
    model.B.assign_sub(learning_rate * db)

# 反复训练，看w和b的变化
# 收集过程中w和b的变化情况
model = Model()

Ws, Bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    Bs.append(model.B.numpy())
    current_loss = loss(model(inputs),output)

    train(model, inputs, output, learning_rate=0.1)
    print('epoch %2d: w=%.2f b=%.2f, loss=%.5f'%(epoch, Ws[-1], Bs[-1], current_loss))


plt.plot(epochs, Ws, 'r',
         epochs, Bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
#右上角标签
plt.legend(['W', 'B', 'TRUE_W', 'TRUE_B'])
plt.show()
