import base_settings
import base_dicom_process
import os
import glob
import random
import ntpath
import numpy
import shutil
# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = base_settings.MEAN_PIXEL_VALUE_NODULE    # 41
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
def get_train_holdout_files(train_percentage=80, manual_labels=True, full_luna_set=False):
    print("Get train/holdout files.")

    # 读取 ndsb3/generated_traindata/luna16_train_cubes_lidc/*.png 图片地址
    # pos_samples 是图片地址列表
    pos_samples = glob.glob(base_settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_lidc/*.png")
    print("Pos samples: ", len(pos_samples))


    # 读取 ndsb3/generated_traindata/luna16_train_cubes_manual/*_pos.png 图片地址
    pos_samples_manual = glob.glob(base_settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_manual/*_pos.png")
    print("Pos samples manual: ", len(pos_samples_manual))
    pos_samples += pos_samples_manual
    print('pos samples:', len(pos_samples))

    # 将列表中元素打乱
    random.shuffle(pos_samples)
    train_pos_count = int((len(pos_samples) * train_percentage) / 100)
    # 目前共有51张图片，pos_samples_train 为40张图片地址（0~39），pos_samples_holdout 为 11 张图片地址(40~50)
    pos_samples_train = pos_samples[:train_pos_count]
    pos_samples_holdout = pos_samples[train_pos_count:]
    if full_luna_set:
        # pos_samples_train 变为所有数据，共51张图片的地址
        pos_samples_train += pos_samples_holdout
        print('pos_samples_train:', len(pos_samples_train))
        if manual_labels:
            pos_samples_holdout = []


    # ndsb3/generated_traindata/ndsb3_train_cubes_manual/*.png
    # 因为目前没有ndsb3数据，所以，以下87~128结果均为0
    ndsb3_list = glob.glob(base_settings.BASE_DIR_SSD + "generated_traindata/ndsb3_train_cubes_manual/*.png")
    print("Ndsb3 samples: ", len(ndsb3_list))


    # ndsb3/generate_traindata/luna16_train_cubes_auto/*_edge.png
    # 数量为好多
    neg_samples_edge = glob.glob(base_settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_edge.png")
    print("Edge samples: ", len(neg_samples_edge))


    # ndsb3/generate_traindata/luna16_train_cubes_auto/*_luna.png
    # 数量为好多
    neg_samples_luna = glob.glob(base_settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_luna.png")
    print("Luna samples: ", len(neg_samples_luna))

    # neg_samples = neg_samples_edge + neg_samples_white

    # neg 总样本集图片地址集合
    neg_samples = neg_samples_edge + neg_samples_luna
    # 打乱顺序
    random.shuffle(neg_samples)

    train_neg_count = int((len(neg_samples) * train_percentage) / 100)

    neg_samples_falsepos = []


    # ndsb3/generated_traindata/luna16_train_cubes_auto/*_falsepos.png
    for file_path in glob.glob(base_settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/*_falsepos.png"):
        neg_samples_falsepos.append(file_path)
    print("Falsepos LUNA count: ", len(neg_samples_falsepos))
    print('luna + edge + falsepos count:', len(neg_samples_falsepos)+len(neg_samples))

    neg_samples_train = neg_samples[:train_neg_count]
    print('neg_samples_train:', len(neg_samples_train))
    # 啥意思，为啥加三遍一样的？
    neg_samples_train += neg_samples_falsepos + neg_samples_falsepos + neg_samples_falsepos
    neg_samples_holdout = neg_samples[train_neg_count:]
    if full_luna_set:
        neg_samples_train += neg_samples_holdout
        print('neg_samples_train = neg_samples_train+3遍 neg_sample_false : ', len(neg_samples_train))

    train_res = []
    holdout_res = []
    sets = [(train_res, pos_samples_train, neg_samples_train), (holdout_res, pos_samples_holdout, neg_samples_holdout)]
    for set_item in sets:
        pos_idx = 0

        # negtative样本数量 和 positive 样本数量比值， NEGS_PER_POS=20
        negs_per_pos = NEGS_PER_POS
        # res 是 第一次：train_res = []   第二次：holdout_res
        res = set_item[0]
        #neg_samples 是 第一次：neg_samples_train     第二次：neg_samples_holdout
        neg_samples = set_item[2]
        # pos_samples 是 第一次：pos_samples_train: 51 张   第二次：pos_samples_holdout
        pos_samples = set_item[1]
        print("Pos", len(pos_samples))
        ndsb3_pos = 0
        ndsb3_neg = 0
        for index, neg_sample_path in enumerate(neg_samples):
            # res.append(sample_path + "/")
            # (ndsb3/generated_traindata/luna16_train_cubes_auto/*.png,0,0)
            res.append((neg_sample_path, 0, 0))
            if index % negs_per_pos == 0:
                # pos_idx 初始为 0，每隔20个 neg 样本，增加一个 pos 样本
                pos_sample_path = pos_samples[pos_idx]
                file_name = ntpath.basename(pos_sample_path)
                # 以 “_” 分割地址，parts是一个结果列表
                # 地址例子： 1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_0_4_1_pos.png
                parts = file_name.split("_")
                if parts[0].startswith("ndsb3manual"):
                    if parts[3] == "pos":
                        class_label = 1  # only take positive examples where we know there was a cancer..
                        cancer_label = int(parts[4])
                        assert cancer_label == 1
                        size_label = int(parts[5])
                        # print(parts[1], size_label)
                        assert class_label == 1
                        if size_label < 1:
                            print("huh ?")
                        assert size_label >= 1
                        ndsb3_pos += 1
                    else:
                        class_label = 0
                        size_label = 0
                        ndsb3_neg += 1
                else:
                    # 以 1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_0_4_1_pos.png 为例
                    # 私人推测 class_label 指是否患癌，size_label 指患癌的程度（源标签程度为1~5，经过加工，变为1~100）
                    class_label = int(parts[-2])    # class_label=1
                    size_label = int(parts[-3])     # size_label=4
                    # 断言函数，如果 assert后跟的条件为 False， 则程序崩溃报错
                    assert class_label == 1
                    assert parts[-1] == "pos.png"
                    assert size_label >= 1

                res.append((pos_sample_path, class_label, size_label))
                pos_idx += 1
                pos_idx %= len(pos_samples)

        print("ndsb2 pos: ", ndsb3_pos)
        print("ndsb2 neg: ", ndsb3_neg)
    # train_res 就是 res
    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res


# 这是一个训练数据和验证数据生成器，每个batch返回一次数据
def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        # class_list 是“是否患癌”的标签集合(0\1)
        class_list = []
        # size_list 是“患癌程度”的标签集合(1~25)
        size_list = []
        if train_set:
            # 打乱训练集顺序
            random.shuffle(record_list)
        CROP_SIZE = CUBE_SIZE       # 32
        # CROP_SIZE = 48

        # 对每张图片进行遍历
        for record_idx, record_item in enumerate(record_list):
            #rint patient_dir
            class_label = record_item[1]
            size_label = record_item[2]
            # 图片是“非患癌”，即 neg 图像集
            if class_label == 0:
                # cube_image : 48*48*48
                cube_image = base_dicom_process.load_cube_img(record_item[0], 6, 8, 48)
                # if train_set:
                #     # helpers.save_cube_img("c:/tmp/pre.png", cube_image, 8, 8)
                #     cube_image = random_rotate_cube_img(cube_image, 0.99, -180, 180)
                #
                # if train_set:
                #     if random.randint(0, 100) > 0.1:
                #         # cube_image = numpy.flipud(cube_image)
                #         cube_image = elastic_transform48(cube_image, 64, 8, random_state)
                wiggle = 48 - CROP_SIZE - 1     # wiggle : 15
                indent_x = 0
                indent_y = 0
                indent_z = 0
                if wiggle > 0:
                    indent_x = random.randint(0, wiggle)
                    indent_y = random.randint(0, wiggle)
                    indent_z = random.randint(0, wiggle)
                # 在48*48*48的立方体中，随机裁剪出32*32*32的小立方体
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]

                if train_set:
                    if random.randint(0, 100) > 50:
                        # 对三维矩阵进行翻转
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        # 另一种翻转
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        # 另一种翻转
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        # 另一种翻转
                        cube_image = cube_image[:, ::-1, :]

                if CROP_SIZE != CUBE_SIZE:
                    cube_image = base_dicom_process.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            # 图像是“患癌”，即 pos 图片集
            else:
                # cube_image : 64*64*64
                cube_image = base_dicom_process.load_cube_img(record_item[0], 8, 8, 64)

                if train_set:
                    pass

                current_cube_size = cube_image.shape[0]            # 64
                indent_x = (current_cube_size - CROP_SIZE) / 2     # 16
                indent_y = (current_cube_size - CROP_SIZE) / 2     # 16
                indent_z = (current_cube_size - CROP_SIZE) / 2     # 16
                wiggle_indent = 0
                wiggle = current_cube_size - CROP_SIZE - 1         # 31
                if wiggle > (CROP_SIZE / 2):
                    wiggle_indent = CROP_SIZE / 4                  # 8
                    wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1   # 15
                if train_set:
                    indent_x = wiggle_indent + random.randint(0, wiggle)
                    indent_y = wiggle_indent + random.randint(0, wiggle)
                    indent_z = wiggle_indent + random.randint(0, wiggle)

                indent_x = int(indent_x)
                indent_y = int(indent_y)
                indent_z = int(indent_z)
                # 在64*64*64的立方体中，随机裁剪出32*32*32的小立方体（这里好像不太随机，小立方体像素范围是（8~54）？）
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
                if CROP_SIZE != CUBE_SIZE:
                    cube_image = base_dicom_process.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

                if train_set: # 以下为四种随机翻转方式
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]

            # cube_image.mean() 计算三维矩阵所有数的平均数，结果是一个数
            means.append(cube_image.mean())
            # cube_image 是 32*32 *32
            # img3d 为 1*32*32*32*1
            img3d = prepare_image_for_net3D(cube_image)
            if train_set:
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            img_list.append(img3d)
            class_list.append(class_label)
            size_list.append(size_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                print('x shape:', x.shape)
                y_class = numpy.vstack(class_list)
                y_size = numpy.vstack(size_list)
                # yield 是返回部分，详见python生成器解释
                yield x, {"out_class": y_class, "out_malignancy": y_size}
                img_list = []
                class_list = []
                size_list = []
                batch_idx = 0


# 函数后面的箭头是对返回值的注释
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
    # 1*1*1*1:一个像素点，即一个值
    out_class = Convolution3D(1, (1, 1, 1), activation="sigmoid", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)
    # 1*1*1*1：一个像素点，即一个值
    out_malignancy = Convolution3D(1, (1, 1, 1), activation=None, name="out_malignancy_last")(last64)
    out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    # 定义一个有一个输入和两个输出的模型
    model = Model(inputs=inputs, outputs=[out_class, out_malignancy])
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    # 定义损失函数、优化函数、和评测方法
    # optimzer:SGD()是随机梯度下降以及对应参数
    # loss:计算损失函数，这里指定了两个损失函数，分别对应两个输出结果，out_class:binary_crossentropy,  out_malignancy:mean_absolute_error
    # metris：性能评估函数,这里指定了两个性能评估函数
    # binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率，binary_crossentropy是对数损失
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error}, metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

    if features:
        model = Model(input=inputs, output=[last64])
    # 打印出模型概况
    model.summary(line_length=140)

    return model


# 以epoch为参数，得到一个新的学习率
def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def train(model_name, fold_count, train_full_set=False, load_weights_path=None, ndsb3_holdout=0, manual_labels=True):
    batch_size = 128

    # train_files 为训练集图片地址（ndsb3/generated_traindata中三个文件夹的所有数据，且含有部分翻倍）
    # holdout_files 为 20% 的 总图片数量（ndsb3/generated_traindata中三个文件夹的 20% 数据， 大概是104708张图片左右）
    # 数据集里的数据存储格式为：(neg_sample_path, 0, 0) 和 (pos_sample_path, class_label, size_label)
    # 第二项指是否患癌，由0，1 两种，第三项为患癌程度，范围为1~25
    # neg_sample_path 是未患癌病人图像地址，所以第二第三项都为0
    # pos_sample_path 是患癌病人的图像地址，所以第二项为1， 第三项为1~25
    train_files, holdout_files = get_train_holdout_files(train_percentage=80, ndsb3_holdout=ndsb3_holdout, manual_labels=manual_labels, full_luna_set=train_full_set, fold_count=fold_count)

    # 取部分训练数据和 holdout数据（验证数据集）
    # train_files = train_files[:100]
    # holdout_files = train_files[:10]
    # train_gen 和 holdout_gen 结构：[x, {"out_class": y_class, "out_malignancy": y_size}]
    train_gen = data_generator(batch_size, train_files, True)
    holdout_gen = data_generator(batch_size, holdout_files, False)
    for i in range(0, 10):
        tmp = next(holdout_gen)
        cube_img = tmp[0][0].reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE
        # helpers.save_cube_img("c:/tmp/img_" + str(i) + ".png", cube_img, 4, 8)
        # print(tmp)

    # keras 的动态学习率调度，learnrate_scheduler是一个新的学习率
    learnrate_scheduler = LearningRateScheduler(step_decay)
    model = get_net(load_weight_path=load_weights_path)
    holdout_txt = "_h" + str(ndsb3_holdout) if manual_labels else ""
    if train_full_set:
        # _fs
        holdout_txt = "_fs" + holdout_txt
    # workdir/model_luna16_full__fs_e
    # 每隔1轮保存一次模型
    checkpoint = ModelCheckpoint("workdir/model_" + model_name + "_" + holdout_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
                                 monitor='val_loss', verbose=1, save_best_only=not train_full_set, save_weights_only=False, mode='auto', period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    checkpoint_fixed_name = ModelCheckpoint("workdir/model_" + model_name + "_" + holdout_txt + "_best.hd5",
                                            monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # 每轮进行1024个batch，共进行100轮
    model.fit_generator(generator=train_gen,steps_per_epoch=int(len(train_files)/batch_size),epochs=10, validation_data=holdout_gen,
                        validation_steps=int(len(holdout_files)/batch_size), callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler])
    model.save("workdir/model_" + model_name + "_" + holdout_txt + "_end.hd5")


if __name__ == "__main__":
    if True:
        # model 1 on luna16 annotations. full set 1 versions for blending
        train(train_full_set=True, load_weights_path=None, model_name="luna16_full", fold_count=-1, manual_labels=False)
        if not os.path.exists("models/"):
            os.mkdir("models")
        shutil.copy("workdir/model_luna16_full__fs_best.hd5", "models/model_luna16_full__fs_best.hd5")

