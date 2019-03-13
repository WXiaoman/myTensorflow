import tensorflow as tf
import numpy

tf.enable_eager_execution()

x = tf.constant(5.0)
# x = tf.ones((2, 2))
# with tf.GradientTape() as g:
#     g.watch(x)
#     n = tf.reduce_sum(x)
#     y = n*n
# dy_dn = g.gradient(y, n)
# print(dy_dn)
# assert dy_dn.numpy() == 8.0
with tf.GradientTape() as g:
    g.watch(x)
    y = x*x
dy_dx = g.gradient(y, x)
print(dy_dx)
assert dy_dx.numpy() == 10.0
