# 对影评分类
# 网页链接https://www.tensorflow.org/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
#num_words=10000表示保留训练数据中出现频率在前10000的字词，为了确定规模的可管理性，罕见字词会被舍弃
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#将单词索映射回单词
word_index = imdb.get_word_index()
#第一个指数保留
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    # 用' '空白格把内容连接起来，从字典中找i，找不到的话返回默认值"?"
    return ' '.join(reverse_word_index.get(i, "?") for i in text)
# 整数数组必须转换为张量才能feed到神经网络中，可以填充数组，让他们都有一样的长度，
# 然后创建一个形状是max_length * num_reviews的整数张量
# 可以用一个能够处理这种形状的嵌入层作为网络中的第一层
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=256
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=256
)

# 以上数据准备完毕，接下来构建网络
# 输入形状是用于影评的词汇数量（10000字）
vocab_size = 10000
model = keras.Sequential()
# 在整数编码的词汇表中找到每个字词-索引的嵌入向量。训练是会学习这些向量，这些向量会向输出数组添加一个维度
# 生成的维度是(batch, sequence, embedding)
model.add(keras.layers.Embedding(vocab_size,16))
# 对序列维度求平均值，对每个样本返回一个长度固定的输出向量。这样模型就可以尽可能简单地处理各种输入长度
model.add(keras.layers.GlobalAveragePooling1D())
# 长度固定的输出向量传入接下来这个全连接层，包含16个隐藏单元
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
# 与输出节点相连。用了sigmoid激活后输出都是0-1的浮点数，表示概率/置信度
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

# 损失函数和优化器.binary_crossentropy可以测量概率之间的差距，在这个例子里是预测的概率和实际分布之间的差距
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 创建验证集。从原始数据中选出10000个样本，作为验证集。
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
# 训练模型
# 用有512个样本的小批次训练模型40个周期。训练期间监控验证集的10000个样本上的损失和准确率
# fit返回了一个history对象，包含一个字典，其中包括训练期间发生的情况
# dict_keys(['loss', 'val_loss', 'val_acc', 'acc'])
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
# 评估模型
results = model.evaluate(test_data, test_labels)

history_dict = history.history
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs,val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

#plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
