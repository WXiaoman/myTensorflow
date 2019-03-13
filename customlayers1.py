#

# https://www.tensorflow.org/tutorials/eager/custom_layers

import tensorflow as tf
tf.enable_eager_execution()

my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(2, 1,
                                                      padding='same'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(3, (1, 1)),
                               tf.keras.layers.BatchNormalization()])
print(my_seq(tf.zeros([1, 2, 3, 3])))
