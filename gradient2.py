# 对微分的计算
# 网页链接  https://www.tensorflow.org/tutorials/eager/automatic_differentiation

import tensorflow as tf
import numpy

tf.enable_eager_execution()

x = tf.Variable(1.0)

with tf.GradientTape() as g:
    with tf.GradientTape() as t:
        y = x*x*x
        dy_dx = t.gradient(y, x)
    d2y_dx = g.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx)
assert dy_dx.numpy() == 3.0
assert d2y_dx.numpy() == 6.0
