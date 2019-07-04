import base_settings
import os
import glob
import random
import numpy

# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten, Dropout, AveragePooling3D, Dense
from keras.models import Model
from keras.metrics import categorical_accuracy, binary_crossentropy, binary_accuracy
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import utils
import shutil
import base_dicom_process

K.clear_session()


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
LEARN_RATE = 0.001
USE_DROPOUT = False
log_dir = './logs/000'
mean_pixel_values = 118


# 得到图像均值，用来进行0值中心化
def get_mean_pixels():
    pixel_value = 0
    print("Get train/holdout files.")
    src_dir = 'D:/Mywork/image_coord_regenerate/dataset/train/'
    one_samples = glob.glob(src_dir+'one/' + "*.png")
    two_samples = glob.glob(src_dir+'two/' + "*.png")
    three_samples = glob.glob(src_dir+'three/' + "*.png")
    four_samples = glob.glob(src_dir+'four/' + "*.png")
    five_samples = glob.glob(src_dir+'zero/' + "*.png")
    # sample_count = [len(one_samples), len(two_samples), len(three_samples), len(four_samples), len(five_samples)]
    # print('sample count is:', sample_count)
    record_item = one_samples+two_samples+three_samples+four_samples+five_samples
    random.shuffle(record_item)
    for item in record_item:
        cube_image = base_dicom_process.load_cube_img(item, 8, 8, 64)
        pixel_value += cube_image.mean()
    print(len(record_item))
    print('pixel_value sum is:', pixel_value)
    print('avg pixel value is:', pixel_value/len(record_item))


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= mean_pixel_values
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


# 生成训练和验证集图像地址列表并返回
def get_train_holdout_files(train_percentage=0.8, full_luna_set=False):
    print("Get train/holdout files.")
    src_dir = 'D:/Mywork/image_coord_regenerate/dataset/train/'
    one_samples = glob.glob(src_dir+'one/' + "*.png")
    two_samples = glob.glob(src_dir+'two/' + "*.png")
    three_samples = glob.glob(src_dir+'three/' + "*.png")
    four_samples = glob.glob(src_dir+'four/' + "*.png")
    five_samples = glob.glob(src_dir+'zero/' + "*.png")
    # sample_count = [len(one_samples), len(two_samples), len(three_samples), len(four_samples), len(five_samples)]
    # print('sample count is:', sample_count)

    # pos_samples 是图片地址列表
    # pos_samples = one_samples+two_samples*4+three_samples*20+four_samples*7+five_samples*23
    # pos_samples = one_samples + two_samples*5 + four_samples*7
    # 将列表中元素打乱
    # random.shuffle(pos_samples)
    # random.shuffle(one_samples)
    # random.shuffle(two_samples)
    # random.shuffle(four_samples)

    one_sample_train = int(len(one_samples) * train_percentage)
    two_sample_train = int(len(two_samples) * train_percentage)
    three_sample_train = int(len(three_samples) * train_percentage)
    four_sample_train = int(len(four_samples) * train_percentage)
    five_sample_train = int(len(five_samples) * train_percentage)
    # print(one_sample_train, two_sample_train, four_sample_train)

    samples_train = one_samples[:one_sample_train] + two_samples[:two_sample_train] + three_samples[:three_sample_train] \
                    + four_samples[:four_sample_train] + five_samples[:five_sample_train]
    samples_holdout = one_samples[one_sample_train:] + two_samples[two_sample_train:] + three_samples[:three_sample_train:] \
                      + four_samples[four_sample_train:] + five_samples[five_sample_train:]
    random.shuffle(samples_train)
    random.shuffle(samples_holdout)


    if full_luna_set:
        # pos_samples_train 变为所有数据
        samples_train += samples_holdout
        print('samples_train:', len(samples_train))


    train_res = []
    holdout_res = []
    for sample in samples_train:
        class_label = sample.split('_')[-2]

        # new_class_label = int(class_label)
        # if new_class_label == 1:
        #     train_res.append([sample, str(0)])
        # elif new_class_label == 2:
        #     train_res.append([sample, str(1)])
        # # elif new_class_label == 4:
        # #     train_res.append([sample, str(2)])
        # else:
        #     assert False, '训练集标签错误'

        # print(class_label)
        if class_label == 'SCLC':
            train_res.append([sample, '0'])
        else:
            train_res.append([sample, class_label])
    # print('**************************************')
    for h_sample in samples_holdout:
        h_class_label = h_sample.split('_')[-2]

        # h_new_class_label = int(h_class_label)
        # if h_new_class_label == 1:
        #     holdout_res.append([h_sample, str(0)])
        # elif h_new_class_label == 2:
        #     holdout_res.append([h_sample, str(1)])
        # # elif h_new_class_label == 4:
        # #     holdout_res.append([h_sample, str(2)])
        # else:
        #     assert False, '验证集标签错误'

        # print(h_class_label)
        if h_class_label == 'SCLC':
            holdout_res.append([h_sample, '0'])
        else:
            holdout_res.append([h_sample, h_class_label])

    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res


# 这是一个训练数据和验证数据生成器，每个batch返回一次数据
def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    means = []
    # random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        # class_list 是“肺癌类型”的标签集合[1，2，3，4, 5]
        class_list = []
        if train_set:
            # 打乱训练集顺序
            random.shuffle(record_list)

        # CROP_SIZE = CUBE_SIZE       # 32

        # 对每张图片进行遍历
        for record_idx, record_item in enumerate(record_list):
            #rint patient_dir
            # print(record_item)
            class_label = int(record_item[1])
            # cube_image : 64*64*64
            cube_image = base_dicom_process.load_cube_img(record_item[0], 8, 8, 64)

            # current_cube_size = cube_image.shape[0]  # 64
            # indent_x = (current_cube_size - CROP_SIZE) / 2  # 16
            # indent_y = (current_cube_size - CROP_SIZE) / 2  # 16
            # indent_z = (current_cube_size - CROP_SIZE) / 2  # 16
            # wiggle_indent = 0
            # wiggle = current_cube_size - CROP_SIZE - 1  # 31
            # if wiggle > (CROP_SIZE / 2):
            #     wiggle_indent = CROP_SIZE / 4  # 8
            #     wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1  # 15
            # if train_set:
            #     indent_x = wiggle_indent + random.randint(0, wiggle)
            #     indent_y = wiggle_indent + random.randint(0, wiggle)
            #     indent_z = wiggle_indent + random.randint(0, wiggle)
            #
            # indent_x = int(indent_x)
            # indent_y = int(indent_y)
            # indent_z = int(indent_z)
            # # 在64*64*64的立方体中，随机裁剪出32*32*32的小立方体（这里好像不太随机，小立方体像素范围是（8~54）
            # cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
            #              indent_x:indent_x + CROP_SIZE]
            # assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

            if train_set:  # 以下为四种随机翻转方式
                if random.randint(0, 100) > 50:
                    cube_image = numpy.fliplr(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = numpy.flipud(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, :, ::-1]
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, ::-1, :]

            # mean_pixel_value = cube_image.mean()  # 计算三维矩阵所有数的平均数，结果是一个数
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
                # yield x, {"lastdense": one_hot_labels}
                img_list = []
                class_list = []
                batch_idx = 0

def get_net(input_shape=(64, 64, 64, 1), load_weight_path=None, features=False) -> Model:
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(strides=(2, 1, 1), pool_size=(2, 1, 1), padding="same")(x)

    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1', )(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)


    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)

    if USE_DROPOUT:
        x = Dropout(rate=0.3)(x)

    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)

    if USE_DROPOUT:
        x = Dropout(rate=0.4)(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)

    if USE_DROPOUT:
        x = Dropout(rate=0.5)(x)

    # 新加部分
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b', strides=(1, 1, 1), )(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)

    last64 = Convolution3D(64, (2, 2, 2), activation="relu", name="last_64")(x)

    out_class = Convolution3D(5, (1, 1, 1), activation="softmax", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)
    # 1*1*1*1：一个像素点，即一个值
    # out_malignancy = Convolution3D(1, (1, 1, 1), activation=None, name="out_malignancy_last")(last64)
    # out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    # 定义一个有一个输入一个输出的模型
    model = Model(inputs=inputs, outputs=out_class)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    # 定义损失函数、优化函数、和评测方法
    # optimzer:SGD()是随机梯度下降以及对应参数
    # loss:计算损失函数，这里指定了两个损失函数，分别对应两个输出结果，out_class:binary_crossentropy,  out_malignancy:mean_absolute_error
    # metris：性能评估函数,这里指定了两个性能评估函数
    # binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率，binary_crossentropy是对数损失
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True),
                  loss=categorical_crossentropy, metrics=[categorical_crossentropy, categorical_accuracy])

    if features:
        model = Model(input=inputs, output=[last64])
    # 打印出模型概况
    model.summary(line_length=140)
    return model

# 函数后面的箭头是对返回值的注释
# 判断肺癌类型网络
def get_vggnet(input_shape=(64, 64, 64, 1), load_weight_path=None, features=False) -> Model:

    inputs = Input(shape=input_shape, name="input_1")
    x = inputs

    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1.1')(x)
    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1.2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool1')(x)


    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2.1', strides=(1, 1, 1))(x)
    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2.2', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)

    # if USE_DROPOUT:
    #     x = Dropout(rate=0.3)(x)

    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3.1', strides=(1, 1, 1))(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3.2', strides=(1, 1, 1))(x)
    x = Convolution3D(256, (1, 1, 1), activation='relu', padding='same', name='conv3.3', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)

    # if USE_DROPOUT:
    #     x = Dropout(rate=0.4)(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4.1', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4.2', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (1, 1, 1), activation='relu', padding='same', name='conv4.3', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)

    # if USE_DROPOUT:
    #     x = Dropout(rate=0.5)(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5.1', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5.2', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (1, 1, 1), activation='relu', padding='same', name='conv5.3', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)

    x = Flatten()(x)

    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    out_class = Dense(6, activation="softmax", name='lastdense')(x)
    # out_class = Convolution3D(6, (1, 1, 1), activation="softmax", name="out_class_last")(last64)
    # out_class = Flatten(name="out_class")(out_class)

    # 定义一个有一个输入、一个输出的模型
    model = Model(inputs=inputs, outputs=out_class)
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    # 定义损失函数、优化函数、和评测方法
    # optimzer:SGD()是随机梯度下降以及对应参数
    # loss:计算损失函数，这里指定了两个损失函数，分别对应两个输出结果，out_class:binary_crossentropy,  out_malignancy:mean_absolute_error
    # metris：性能评估函数,这里指定了两个性能评估函数
    # binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率，binary_crossentropy是对数损失
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True),
                  loss=categorical_crossentropy, metrics=[categorical_crossentropy, categorical_accuracy])

    if features:
        model = Model(input=inputs, output=[out_class])
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


# 自定义损失函数，总召回率
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives/possible_positives
    return recall

def train(model_name, train_full_set=False, load_weights_path=None):
    batch_size = 8

    # train_files 为训练集图片地址（D:\Mywork\data\generated_traindata，且含有部分翻倍）
    # holdout_files 为 20% 的 总图片数量
    # train_files 和 holdout_files 的数据存储格式为：(sample_path, class_type)
    # 第二项指肺癌类型，[1,2,3,4]
    train_files, holdout_files = get_train_holdout_files(train_percentage=0.9, full_luna_set=train_full_set)

    # 生成训练和验证batch
    # train_gen 和 holdout_gen 结构：[x, {"out_class": y_class}]
    train_gen = data_generator(batch_size, train_files, True)
    holdout_gen = data_generator(batch_size, holdout_files, False)

    # for i in range(0, 10):
    #     tmp = next(holdout_gen)
    #     cube_img = tmp[0][0].reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
    #     cube_img = cube_img[:, :, :, 0]
    #     cube_img *= 255.
    #     cube_img += MEAN_PIXEL_VALUE


    # keras 的动态学习率调度，learnrate_scheduler是一个新的学习率
    learnrate_scheduler = LearningRateScheduler(step_decay)
    model = get_net(load_weight_path=load_weights_path)

    # 每隔1轮保存一次模型
    checkpoint = ModelCheckpoint("workdir/cancertype/model_" + model_name + "_" + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    checkpoint_fixed_name = ModelCheckpoint("workdir/cancertype/model_" + model_name + "_best.hd5",
                                            monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    model.fit_generator(generator=train_gen, steps_per_epoch=int(len(train_files)/batch_size), epochs=1000, validation_data=holdout_gen,
                        validation_steps=int(len(holdout_files)/batch_size), callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler, TensorBoard(log_dir=log_dir)])

    # predIdxs = model.predict_generator(holdout_gen, steps=int(len(holdout_files)/batch_size))
    # print('predIdxs:', predIdxs)

    model.save("workdir/cancertype/model_" + model_name + "_end.hd5")


if __name__ == "__main__":
    # 计算图像均值
    # get_mean_pixels()
    model_path = "models/model_cancer_type__fs_best.hd5"
    if True:
        train(train_full_set=False, load_weights_path=None, model_name="cancer_type")
        if not os.path.exists("models/"):
            os.mkdir("models")
        shutil.copy("workdir/cancertype/model_cancer_type_best.hd5", "models/model_cancer_type_best.hd5")


