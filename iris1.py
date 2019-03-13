# 对鸢尾花数据集分类
# https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough

from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# print("TensorFlow version: {}".format(tf.VERSION))
# print("Eager execution: {}".format(tf.executing_eagerly()))
train_database_url = "http://download.tensorflow.org/data/iris_training.csv"
train_database_fp = tf.keras.utils.get_file(fname=os.path.basename(train_database_url),
                                            origin=train_database_url)
# print(train_database_fp)
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
# print(feature_names)
# print(label_name)
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(
    train_database_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)
# features, labels = next(iter(train_dataset))

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()),axis=1)
    return features, labels
train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
print(features[:5])
'''
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
'''

# print(features[:5])
# tf.keras.Sequential 模型是层的线性堆叠。
# 该模型的构造函数会采用一系列层实例；
# 在本示例中，采用的是 2 个密集层（分别包含 10 个节点）以及 1 个输出层（包含 3 个代表标签预测的节点）。
# 第一个层的 input_shape 参数对应该数据集中的特征数量，它是一项必需参数。

# 激活函数可决定层中每个节点的输出形状。
# 这些非线性关系很重要，如果没有它们，模型将等同于单个层。激活函数有很多，但隐藏层通常使用 ReLU。
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # 必须有input_shape
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])
predictions = model(features)
predictions[:5]
prediction = tf.argmax(predictions, axis=1)
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

# 定义损失和梯度函数
def loss(model, x, y):
    y_ = model(x)
    # 使用 tf.losses.sparse_softmax_cross_entropy 函数计算其损失，
    # 此函数会接受模型的类别概率预测结果和预期标签，然后返回样本的平均损失。
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# 设置优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
global_step = tf.train.get_or_create_global_step()
loss_value, grads = grad(model, features, labels)
optimizer.apply_gradients(zip(grads, model.variables), global_step)
# 训练数据
## Note: Rerunning this cell uses the same model variables
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12,8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel('loss', fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('accuracy', fontsize=14)
axes[1].set_xlabel('epoch', fontsize=14)
axes[1].plot(train_accuracy_results)

# 测试
test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
test_dataset = tf.data.experimental.make_csv_dataset(
    train_database_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False
)
test_dataset = test_dataset.map(pack_features_vector)
test_accuracy = tfe.metrics.Accuracy()
for x, y in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("the set accuracy: {:.3%}".format(test_accuracy.result()))

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])
predictionss = model(predict_dataset)

for m,n in enumerate(predictionss):
    idx = tf.argmax(n).numpy()
    p = tf.nn.softmax(n)[idx]
    name = class_names[idx]
    print("example {} prediction: {} ({:4.1f}%)".format(m, name, 100*p))

