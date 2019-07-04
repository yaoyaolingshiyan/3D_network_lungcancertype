import base_settings
import base_dicom_process
import os
import pandas
import numpy
from keras import backend as K
import shutil
import step4_1_train_lungdetector
# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import step4_2_train_typedetector
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# 改变图像维度顺序为tensorflow维度顺序（height，width，channels）
K.set_image_dim_ordering("tf")
CUBE_SIZE = step4_2_train_typedetector.CUBE_SIZE    # 32
MEAN_PIXEL_VALUE = base_settings.MEAN_PIXEL_VALUE_NODULE  # 41
P_TH = 0.6
PREDICT_STEP = 12
USE_DROPOUT = False

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

def predict_cubes(model_path, continue_job, only_patient_id=None, magnification=1, flip=False):

    dst_dir = base_settings.PREDICT_CUBE
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    flip_ext = ""
    if flip:
        flip_ext = "_flip"

    # D:/Mywork/data/predict_cube/predictions10_luna16_fs/
    # dst_dir += "predictions" + str(int(magnification * 10)) + flip_ext + "_" + ext_name + "/"
    # print('dst_dir:', dst_dir)
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    # 开始计时
    sw = base_settings.Stopwatch.start_new()
    # 导入模型
    model = step4_1_train_lungdetector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)

    patient_ids = []

    # 导入数据
    # 这里控制病人数目，只输入一个病人的话，只有一个预测结果
    for file_name in os.listdir(base_settings.DICOM_EXTRACT_DIR):
        if not os.path.isdir(base_settings.DICOM_EXTRACT_DIR + file_name):
            continue
        # 所有的病人文件夹地址
        patient_ids.append(file_name)
        # print(file_name)

    all_predictions_csv = []
    # print('共有病人数目：', len(patient_ids))
    for patient_index, patient_id in enumerate(reversed(patient_ids)):

        if only_patient_id is not None and only_patient_id != patient_id:
            continue

        print('判别病人：', patient_index, ": ", patient_id)
        # D:/Mywork/data/predict_cube/predictions10_luna16_fs/patientID.csv
        # 要生成的结果csv
        csv_target_path = dst_dir + patient_id + ".csv"
        if continue_job and only_patient_id is None:
            if os.path.exists(csv_target_path):
                os.remove(csv_target_path)

        patient_img = base_dicom_process.load_patient_images(patient_id, base_settings.DICOM_EXTRACT_DIR, "*_i.png", [])

        # magnification 是放大倍数
        if magnification != 1:
            patient_img = base_dicom_process.rescale_patient_images(patient_img, (1, 1, 1), magnification)

        patient_mask = base_dicom_process.load_patient_images(patient_id, base_settings.DICOM_EXTRACT_DIR, "*_m.png", [])
        if magnification != 1:
            patient_mask = base_dicom_process.rescale_patient_images(patient_mask, (1, 1, 1), magnification,
                                                                     is_mask_image=True)

            # patient_img = patient_img[:, ::-1, :]
            # patient_mask = patient_mask[:, ::-1, :]

        step = PREDICT_STEP  # 12
        CROP_SIZE = CUBE_SIZE  # 32

        print('patient_img.shape is:', patient_img.shape)

        # predict_volume_shape_list 是以 crop_size为块大小，step为步长，在原立方体上截取cube，可以截取的个数
        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:  # zxy
                predict_volume_shape_list[dim] += 1
                dim_indent += step
        print('predict_volume_shape_list is :', predict_volume_shape_list)
        predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        print("Predict volume shape: ", predict_volume.shape)
        done_count = 0
        skipped_count = 0
        batch_size = 16
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        annotation_index = 0
        for z in range(0, predict_volume_shape[0]):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    cube_img = patient_img[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    # 所有像素点像素值相加
                    if cube_mask.sum() < 2000:
                        skipped_count += 1
                    else:
                        if flip:
                            cube_img = cube_img[:, :, ::-1]

                        # img_prep.shape is (1,32,32,32,1)
                        img_prep = prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:

                            # batch_data is (batch_size,32,32,32,1)
                            batch_data = numpy.vstack(batch_list)

                            # 模型预测函数，该篇核心
                            p = model.predict(batch_data, batch_size=batch_size)
                            print('*********************************')
                            print(p)
                            print('*********************************')

                            for i in range(len(p[0])):
                                # 获得对应cube的zxy
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                # 获得预测的类别数值
                                nodule_chance = p[0][i][0]
                                print(nodule_chance)

                                # 将预测结果存储在predict_volume中(27,25,15)
                                predict_volume[p_z, p_y, p_x] = nodule_chance
                                # print('predict_volume[p_z, p_y, p_x] :', predict_volume[p_z, p_y, p_x])
                                # 当患癌可能性大于0.6时，保存坐标
                                if nodule_chance > P_TH:      # P_TH = 0.6
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2
                                    print('(p_x, p_y, p_z):', p_x, p_y, p_z)
                                    p_z_perc = round(p_z / patient_img.shape[0], 4)
                                    p_y_perc = round(p_y / patient_img.shape[1], 4)
                                    p_x_perc = round(p_x / patient_img.shape[2], 4)
                                    print('(p_x_perc, p_y_perc, p_z_perc):', p_x_perc, p_y_perc, p_z_perc)
                                    patient_predictions_csv_line = [annotation_index, patient_id, p_x_perc, p_y_perc, p_z_perc, nodule_chance]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    # all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1

                            batch_list = []
                            batch_list_coords = []

                    done_count += 1
                    # print('done_count:', done_count)
                    # if done_count % 10000 == 0:
                    #     # done_count 是已经进行了多少个cube（包括测试的和跳过的），skipped是跳过多少个
                    #     print("Done: ", done_count, " skipped:", skipped_count)
        # 创建二维表，第一个参数是数据，第二个参数是列名
        df = pandas.DataFrame(patient_predictions_csv, columns=["anno_index", "patient_id", "coord_x", "coord_y", "coord_z", "nodule_chance"])
        # 将二维表保存为csv文件，第一个参数为地址，index为是否保存索引
        df.to_csv(csv_target_path, index=False)

        # 对 predict_volume 结果统计（患癌类型)
        # result_list = sort_predict(predict_volume)

        print('result is : ', predict_volume.mean())
        # 测试花费时间
        print("Done in : ", sw.get_elapsed_seconds(), " seconds")


if __name__ == "__main__":

    CONTINUE_JOB = True
    dst_dir = 'D:/Mywork/data/predict_cube/'
    only_patient_id = '3107'  # D:/Mywork/data/src_dicom/3107

    # 移除之前针对该病人的预测结果
    if not CONTINUE_JOB or only_patient_id is not None:
        shutil.rmtree('D:/Mywork/data/predict_cube', ignore_errors=True)

    if True:
        predict_cubes("cancer_models/model_luna16_full__best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=1, flip=False)
