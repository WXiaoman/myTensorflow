# https://www.tensorflow.org/tutorials/images/hub_with_keras

# 失败
# 原因：数据集下载失败；从keras导入layers失败

from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

# from tensorflow.keras import layers
# 上一行的导入方法报错：ModuleNotFoundError: No module named 'tensorflow.keras'
from tensorflow.python.keras import layers

# 这个URL无效
data_root = tf.keras.utils.get_file(
  'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

# 用ImageDataGenerator的rescale实现图像模块的期望浮动输入在0-1之间
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))
for image_batch, label_batch in image_data:
    print("image batch shape: ", image_batch.shape)
    print("label batch shape: ", label_batch.shape)
    break

# 下载分类器
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
# 用hub.module加载mobilenet，并用tf.keras.layers.Lambda包裹起来作为keras层
def classifier(x):
    classifier_module = hub.Module(classifier_url)
    return classifier_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))
# classifier_layer = tf.keras.layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()
