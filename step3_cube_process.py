import base_settings
import base_dicom_process
import glob
import pandas
import ntpath
import numpy
import cv2
import os
import shutil

CUBE_IMGTYPE_SRC = "_i"

# 将cube逐个排列在一起保存为图片
def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)

# 依据肿瘤的下，(x,y,z)坐标，以该座标为中心，裁剪出64*64*64的cube
# x,y,z可分别用图片的第（x,y）像素点，第z张图表示（前提是把原CT图像素间距调整好）
def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y
                + block_size, start_x:start_x + block_size]
    return res


def make_annotation_images():
    src_dir = base_settings.TRAIN_LABEL
    dst_dir = base_settings.TRAIN_DATA
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    # 清空该目录
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annotations.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annotations.csv", "")
        df_annos = pandas.read_csv(csv_file)
        print(patient_id, 'len(df_annos): ', len(df_annos))
        if len(df_annos) == 0:
            continue
        # 从 extracted_image 读取图像并将一个病人的图像合在一起，例如：349张330*330的图像，返回一个（349，330，330）的三维矩阵（z,y,x）
        images = base_dicom_process.load_patient_images(patient_id, base_settings.DICOM_EXTRACT_DIR, "*" + CUBE_IMGTYPE_SRC + ".png")
        if images is None:
            continue
        print('images shape is:', images.shape)
        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"])
            coord_y = int(row["coord_y"])
            coord_z = int(row["coord_z"])
            anno_index = int(row["anno_index"])
            print('coord_x:', coord_x)
            print('coord_y:', coord_y)
            print('coord_z:', coord_z)
            print('anno_index:', anno_index)
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + '_' + str(index) + ".png", cube_img, 8, 8)

# 统计各种类数量
def count_type_quantity(src_dir):
    count_list = [0, 0, 0, 0, 0, 0]
    samples = os.listdir(src_dir)
    for name in samples:
        sample_type = str(name.split('_')[1])
        if sample_type in ['1', '2', '3', '4', '5']:
            count_list[int(sample_type)] += 1
    print(count_list)
    print(len(samples))
    print(samples)
    return count_list

# 按比例分离训练集和测试集
def depart_train_test(src_dir, percentage):

    dst_dir = 'D:/Mywork/data/generated_testdata/'
    count_list = glob.glob(src_dir+'*.png')
    quantity = len(count_list)
    print(quantity)
    test_quantity = int(quantity * percentage)
    for i in count_list[0:test_quantity]:
        sample = i.split('\\')[-1]
        print(sample)
        shutil.copyfile(i, dst_dir+sample)
    print('test data saved!')

# 不同种类数据集分开存储
def depart_save_type(src_dir):
    samples = os.listdir(src_dir)
    for name in samples:
        sample_type = str(name.split('_')[1])
        if sample_type in ['1', '2', '3', '4', '5']:
            if sample_type == '1':
                shutil.copyfile(src_dir+name, 'D:/Mywork/data/different_typedata/one/'+name)
            elif sample_type == '2':
                shutil.copyfile(src_dir + name, 'D:/Mywork/data/different_typedata/two/' + name)
            elif sample_type == '3':
                shutil.copyfile(src_dir + name, 'D:/Mywork/data/different_typedata/three/' + name)
            elif sample_type == '4':
                shutil.copyfile(src_dir + name, 'D:/Mywork/data/different_typedata/four/' + name)
            elif sample_type == '5':
                shutil.copyfile(src_dir + name, 'D:/Mywork/data/different_typedata/five/' + name)
            else:
                print('It is Unqualified!')
        print(name, 'is over!')


if __name__ == "__main__":
    src_dir = 'D:/Mywork/data/generated_traindata/'
    # 创建训练数据集地址
    # if not os.path.exists("D:/Mywork/data/generated_traindata/"):
    #     os.mkdir("D:/Mywork/data/generated_traindata/")

    # 创建测试集地址
    # if not os.path.exists("D:/Mywork/data/generated_testdata/"):
    #     os.mkdir("D:/Mywork/data/generated_testdata/")

    # 生成训练数据集
    # make_annotation_images()

    # 统计数据集不同种类数量
    # count_type_quantity(src_dir)

    # 不同种类数据集分开存储
    # depart_save_type(src_dir)

    # 分离训练集和测试集
    depart_train_test('D:/Mywork/data/different_typedata/five/', percentage=0.2)







