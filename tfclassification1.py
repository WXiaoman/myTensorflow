#基本分类-图像分类

import tensorflow as tf
import os
#os.environ[KMP_DUPLICATE_LIB_OK="TRUE"]
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

#查看图片，在一张图上放了很多个图
# plt.figure(figsize=(10, 10))
# for i in range(25):
    #把所有的放成5行5列，当前位置是i+1
    # plt.subplot(5, 5, i+1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # plt.imshow(train_images[i], cmap=plt.cm.binary)
    # plt.xlabel(class_names[train_labels[i]])
#plt.show()

#搭建起模型的各个层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#编译模型
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
#train the model
model.fit(train_images,train_labels,epochs=5)
#模型评估
test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Test accuracy: ", test_acc)
predictions = model.predict(test_images)
#第一张图的预测结果
# print(predictions[0])
#从对每一种的预测置信度中选出最有可能的一个
# print(np.argmax(predictions[0]))
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}{:2.0f}%({})".format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# i=0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
#plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
#预测单个图像
img = test_images[0]
#print(img.shape)

#tf.keras已经优化，可以批次预测，但是即使是使用单个图像，还要先添加到列表中
img = (np.expand_dims(img, 0))
#接下来开始预测
predictions_single = model.predict(img)
print(predictions_single)

# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
print(np.argmax(predictions_single[0]))
