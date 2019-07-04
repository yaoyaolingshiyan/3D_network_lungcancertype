import base_settings
import base_dicom_process
import random
import numpy
from keras import backend as K


# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import step4_2_train_typedetector

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# 改变图像维度顺序为tensorflow维度顺序（height，width，channels）
K.set_image_dim_ordering("tf")
CUBE_SIZE = 64
MEAN_PIXEL_VALUE = 118
P_TH = 0.6
PREDICT_STEP = 12
USE_DROPOUT = False
CROP_SIZE = step4_2_train_typedetector.CUBE_SIZE

# 对结果矩阵进行统计，得到不同结果数量
def sort_predict(test_list):
    a = list(test_list.flatten())
    a_list = list(set(a))
    re_list = []
    for i in a_list:
        counts = a.count(i)
        print(counts)
        re_list.append([i, counts])
    return re_list

def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img

def predict_cubes(model_path, cube_dir):

    # 开始计时
    sw = base_settings.Stopwatch.start_new()
    # 导入模型
    model = step4_2_train_typedetector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1),
                                               load_weight_path=model_path)

    cube_image = base_dicom_process.load_cube_img(cube_dir, 8, 8, 64)
    current_cube_size = cube_image.shape[0]  # 64
    indent_x = (current_cube_size - CROP_SIZE) / 2  # 16
    indent_y = (current_cube_size - CROP_SIZE) / 2  # 16
    indent_z = (current_cube_size - CROP_SIZE) / 2  # 16
    wiggle_indent = 0
    wiggle = current_cube_size - CROP_SIZE - 1  # 31
    if wiggle > (CROP_SIZE / 2):
        wiggle_indent = CROP_SIZE / 4  # 8
        wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1  # 15
    indent_x = wiggle_indent + random.randint(0, wiggle)
    indent_y = wiggle_indent + random.randint(0, wiggle)
    indent_z = wiggle_indent + random.randint(0, wiggle)

    indent_x = int(indent_x)
    indent_y = int(indent_y)
    indent_z = int(indent_z)
    # 在64*64*64的立方体中，随机裁剪出32*32*32的小立方体（这里好像不太随机，小立方体像素范围是（8~54）
    cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                 indent_x:indent_x + CROP_SIZE]
    # print('cube image shape is :', cube_image.shape)
    img_prep = prepare_image_for_net3D(cube_image)
    # batch_list.append(img_prep)
    # 模型预测函数，该篇核心
    p = model.predict(img_prep)
    print('预测结果数：', len(p), p.shape)
    print('*********************************')
    print(p)
    print(p[0].tolist())
    print('the predict lung cancer type is : ', p[0].tolist().index(max(p[0].tolist())))
    # 测试花费时间
    print("Done in : ", sw.get_elapsed_seconds(), " seconds")


if __name__ == "__main__":

    if True:
        # 1、将该原CT集重采样并保存
        # 2、读取坐标并在提取图像集中切割64*64*64图像块，并保存
        # 3、src_dir 是 64*64*64图像保存地址
        src_dir = 'D:/Mywork/data/generated_traindata/5976_2_0.png'
        predict_cubes("models/model_cancer_type__fs_best.hd5", cube_dir=src_dir)
