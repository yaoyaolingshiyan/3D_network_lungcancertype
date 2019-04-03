import settings
import dicom_basic_process
import sys
import os
import glob
import random
import pandas
import ntpath
import cv2
import numpy
from typing import List, Tuple

# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras import utils
from keras.utils import plot_model
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil
import dicom_basic_process


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE    # 41
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
# POS_IMG_DIR = "luna16_train_cubes_pos"
LEARN_RATE = 0.001

USE_DROPOUT = False

def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    # 每个像素点的值都减20
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img

# 生成训练和验证集图像地址列表并返回
def get_train_holdout_files(fold_count, train_percentage=80, logreg=True, ndsb3_holdout=0, manual_labels=True, full_luna_set=False):
    print("Get train/holdout files.")
    # pos_samples = glob.glob(settings.BASE_DIR_SSD + "luna16_train_cubes_pos/*.png")


    # 读取 ndsb3/generated_traindata/luna16_train_cubes_lidc/*.png 图片地址
    # pos_samples 是图片地址列表
    pos_samples = glob.glob(settings.TRAIN_DATA + "*.png")
    print(pos_samples)
    print("samples count : ", len(pos_samples))


    # 将列表中元素打乱
    random.shuffle(pos_samples)
    train_pos_count = int((len(pos_samples) * train_percentage) / 100)
    samples_train = pos_samples[:train_pos_count]
    samples_holdout = pos_samples[train_pos_count:]
    if full_luna_set:
        # pos_samples_train 变为所有数据，共138张图片的地址
        samples_train += samples_holdout
        print('samples_train:', len(samples_train))


    train_res = []
    holdout_res = []
    for sample in samples_train:
        class_label = sample.split('_')[-2].split('.')[0]
        # print(class_label)
        train_res.append([sample, class_label])
    # print('**************************************')
    for h_sample in samples_holdout:
        h_class_label = h_sample.split('_')[-2].split('.')[0]
        # print(h_class_label)
        holdout_res.append([h_sample, h_class_label])

    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res

# 这是一个训练数据和验证数据生成器，每个batch返回一次数据
def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        # class_list 是“肺癌类型”的标签集合[1，2，3，4]
        class_list = []
        if train_set:
            # 打乱训练集顺序
            random.shuffle(record_list)

        CROP_SIZE = CUBE_SIZE       # 32

        # 对每张图片进行遍历
        for record_idx, record_item in enumerate(record_list):
            #rint patient_dir
            # print(record_item)
            class_label = int(record_item[1])
            # cube_image : 64*64*64
            cube_image = dicom_basic_process.load_cube_img(record_item[0], 8, 8, 64)

            current_cube_size = cube_image.shape[0]  # 64
            indent_x = (current_cube_size - CROP_SIZE) / 2  # 16
            indent_y = (current_cube_size - CROP_SIZE) / 2  # 16
            indent_z = (current_cube_size - CROP_SIZE) / 2  # 16
            wiggle_indent = 0
            wiggle = current_cube_size - CROP_SIZE - 1  # 31
            if wiggle > (CROP_SIZE / 2):
                wiggle_indent = CROP_SIZE / 4  # 8
                wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1  # 15
            if train_set:
                indent_x = wiggle_indent + random.randint(0, wiggle)
                indent_y = wiggle_indent + random.randint(0, wiggle)
                indent_z = wiggle_indent + random.randint(0, wiggle)

            indent_x = int(indent_x)
            indent_y = int(indent_y)
            indent_z = int(indent_z)
            # 在64*64*64的立方体中，随机裁剪出32*32*32的小立方体（这里好像不太随机，小立方体像素范围是（8~54）
            cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                         indent_x:indent_x + CROP_SIZE]
            assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

            if train_set:  # 以下为四种随机翻转方式
                if random.randint(0, 100) > 50:
                    cube_image = numpy.fliplr(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = numpy.flipud(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, :, ::-1]
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, ::-1, :]

            # cube_image.mean() 计算三维矩阵所有数的平均数，结果是一个数
            # means.append(cube_image.mean())
            # cube_image 是 32*32 *32
            # img3d 为 1*32*32*32*1
            img3d = prepare_image_for_net3D(cube_image)
            img_list.append(img3d)
            class_list.append(class_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                # print('x shape:', x.shape)
                # print(class_list)
                y_class = numpy.vstack(class_list)
                # 将标签转换为分类的 one-hot 编码
                one_hot_labels = utils.to_categorical(y_class, num_classes=5)
                # print(one_hot_labels)
                # yield 是返回部分，详见python生成器解释
                yield x, {"out_class": one_hot_labels}
                img_list = []
                class_list = []
                batch_idx = 0

# 函数后面的箭头是对返回值的注释
# 判断是否患癌网络
def get_cancer_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False) -> Model:
    inputs = Input(shape=input_shape, name="input_1")
    # 32*32*32*1
    x = inputs
    # 16*32*32*1
    x = AveragePooling3D(strides=(2, 1, 1), pool_size=(2, 1, 1), padding="same")(x)
    # 16*32*32*64
    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1', )(x)
    # 16*16*16*64
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

    # 2nd layer group
    # 16*16*16*128
    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    # 8*8*8*128
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)

    # 3rd layer group
    # conv3a / conv3b 选择一个即可
    # 8*8*8*256
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    # 8*8*8*256
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    # 4*4*4*256
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.4)(x)

    # 4th layer group
    # 4*4*4*512
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    # 4*4*4*512
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    2*2*2*512
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)
    # 1*1*1*64
    last64 = Convolution3D(64, (2, 2, 2), activation="relu", name="last_64")(x)
    # 1*1*1*1:一个像素点，即一个值
    out_class = Convolution3D(1, (1, 1, 1), activation="sigmoid", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)

    # 定义一个有一个输入和一个输出的模型
    model = Model(inputs=inputs, outputs=out_class)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    # 定义损失函数、优化函数、和评测方法
    # optimzer:SGD()是随机梯度下降以及对应参数
    # loss:计算损失函数，这里指定了两个损失函数，分别对应两个输出结果，out_class:binary_crossentropy,  out_malignancy:mean_absolute_error
    # metris：性能评估函数,这里指定了两个性能评估函数
    # binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率，binary_crossentropy是对数损失
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_class": "binary_crossentropy"}, metrics={"out_class": [binary_accuracy, binary_crossentropy]})

    if features:
        model = Model(input=inputs, output=[last64])
    # 打印出模型概况
    model.summary(line_length=140)

    return model


# 函数后面的箭头是对返回值的注释
# 判断肺癌类型网络
def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False) -> Model:
    inputs = Input(shape=input_shape, name="input_1")
    # 32*32*32*1
    x = inputs
    # 16*32*32*1
    x = AveragePooling3D(strides=(2, 1, 1), pool_size=(2, 1, 1), padding="same")(x)
    # 16*32*32*64
    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1', )(x)
    # 16*16*16*64
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

    # 2nd layer group
    # 16*16*16*128
    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    # 8*8*8*128
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)

    # 3rd layer group
    # conv3a / conv3b 选择一个即可
    # 8*8*8*256
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    # 8*8*8*256
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    # 4*4*4*256
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.4)(x)

    # 4th layer group
    # 4*4*4*512
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    # 4*4*4*512
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    # 2*2*2*512
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)
    # 1*1*1*64
    last64 = Convolution3D(64, (2, 2, 2), activation="relu", name="last_64")(x)
    # 1*1*1*5:一个像素点，五种结果
    out_class = Convolution3D(5, (1, 1, 1), activation="softmax", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)
    # 1*1*1*1：一个像素点，即一个值
    # out_malignancy = Convolution3D(1, (1, 1, 1), activation=None, name="out_malignancy_last")(last64)
    # out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    # 定义一个有一个输入和两个输出的模型
    model = Model(inputs=inputs, outputs=out_class)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    # 定义损失函数、优化函数、和评测方法
    # optimzer:SGD()是随机梯度下降以及对应参数
    # loss:计算损失函数，这里指定了两个损失函数，分别对应两个输出结果，out_class:binary_crossentropy,  out_malignancy:mean_absolute_error
    # metris：性能评估函数,这里指定了两个性能评估函数
    # binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率，binary_crossentropy是对数损失
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    if features:
        model = Model(input=inputs, output=[last64])
    # 打印出模型概况
    model.summary(line_length=140)
    plot_model(model, to_file='model.png')

    return model


# 以epoch为参数，得到一个新的学习率
def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def train(model_name, fold_count, train_full_set=False, load_weights_path=None, ndsb3_holdout=0, manual_labels=True):
    batch_size = 16

    # train_files 为训练集图片地址（D:\Mywork\data\generated_traindata，且含有部分翻倍）
    # holdout_files 为 20% 的 总图片数量
    # train_files 和 holdout_files 的数据存储格式为：(sample_path, class_type)
    # 第二项指肺癌类型，[1,2,3,4]
    train_files, holdout_files = get_train_holdout_files(train_percentage=80, ndsb3_holdout=ndsb3_holdout, manual_labels=manual_labels, full_luna_set=train_full_set, fold_count=fold_count)

    # 生成训练和验证batch
    # train_gen 和 holdout_gen 结构：[x, {"out_class": y_class}]
    train_gen = data_generator(batch_size, train_files, True)
    holdout_gen = data_generator(batch_size, holdout_files, False)

    for i in range(0, 10):
        tmp = next(holdout_gen)
        cube_img = tmp[0][0].reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE


    # keras 的动态学习率调度，learnrate_scheduler是一个新的学习率
    learnrate_scheduler = LearningRateScheduler(step_decay)
    model = get_net(load_weight_path=load_weights_path)
    holdout_txt = "_h" + str(ndsb3_holdout) if manual_labels else ""
    if train_full_set:
        # _fs
        holdout_txt = "_fs" + holdout_txt
    # workdir/model_luna16_full__fs_e
    # 每隔1轮保存一次模型
    checkpoint = ModelCheckpoint("workdir/model_" + model_name + "_" + holdout_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=not train_full_set, save_weights_only=False, mode='auto', period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    checkpoint_fixed_name = ModelCheckpoint("workdir/model_" + model_name + "_" + holdout_txt + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(generator=train_gen, steps_per_epoch=int(len(train_files)/batch_size), epochs=10, validation_data=holdout_gen, validation_steps=int(len(holdout_files)/batch_size), callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler])
    model.save("workdir/model_" + model_name + "_" + holdout_txt + "_end.hd5")


if __name__ == "__main__":
    if True:
        train(train_full_set=True, load_weights_path=None, model_name="luna16_full", fold_count=-1, manual_labels=False)
        if not os.path.exists("models/"):
            os.mkdir("models")
        shutil.copy("workdir/model_luna16_full__fs_best.hd5", "models/model_luna16_full__fs_best.hd5")


