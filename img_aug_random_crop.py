import tensorflow as tf
import os
import random

source_file = "/Users/wangxiaoman/Desktop/research/imgcrop-master/example/train/"  # 原始文件地址
target_file = "/Users/wangxiaoman/Desktop/research/imgcrop-master/example/train_aug_crop/"  # 修改后的文件地址
num = 50  # 产生图片次数

if not os.path.exists(target_file):  # 如果不存在target_file，则创造一个
    os.makedirs(target_file)

file_list = os.listdir(source_file)  # 读取原始文件的路径

with tf.Session() as sess:
    for i in range(num):
        max_random = len(file_list) - 1
        a = random.randint(1, max_random)  # 随机数字区间
        image_raw_data = tf.gfile.FastGFile(source_file + file_list[a], "rb").read()  # 读取图片
        print("processing：", file_list[a])
        image_data = tf.image.decode_jpeg(image_raw_data)
        cropped = tf.random_crop(image_data,[280, 280, 1])

        # filpped_le_re = tf.image.random_flip_left_right(image_data)  # 随机左右翻转

        # filpped_up_down = tf.image.random_flip_up_down(image_data)  # 随机上下翻转
        #
        # adjust = tf.image.random_brightness(filpped_up_down, 0.4)  # 随机调整亮度
        #
        # image_data = tf.image.convert_image_dtype(adjust, dtype=tf.uint8)
        image_data = tf.image.convert_image_dtype(cropped, dtype=tf.uint8)

        encode_data = tf.image.encode_jpeg(image_data)

        # for i in range(max_random):
        #     file_list[i] = file_list[i].split('.')[0]

        with tf.gfile.GFile(target_file + file_list[a].split('.')[0] + "_crop" +str(i) + ".jpeg", "wb") as f:
            f.write(encode_data.eval())
print("crop is over")
