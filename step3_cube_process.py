import settings
import dicom_basic_process
import glob
import pandas
import ntpath
import numpy
import cv2
import os

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
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def make_annotation_images():
    src_dir = settings.TRAIN_LABEL
    dst_dir = settings.TRAIN_DATA
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    # 清空该目录
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    #
    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annotations.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annotations.csv", "")
        df_annos = pandas.read_csv(csv_file)
        print(patient_id, 'len(df_annos): ', len(df_annos))
        if len(df_annos) == 0:
            continue
        # 从 extracted_image 读取图像并将一个病人的图像合在一起，例如：349张330*330的图像，返回一个（349，330，330）的三维矩阵（z,y,x）
        images = dicom_basic_process.load_patient_images(patient_id, settings.DICOM_EXTRACT_DIR, "*" + CUBE_IMGTYPE_SRC + ".png")
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


if __name__ == "__main__":
    # 创建训练数据集地址
    if not os.path.exists("D:/Mywork/data/generated_traindata/"):
        os.mkdir("D:/Mywork/data/generated_traindata/")

    if True:
        # 生成训练数据集
        make_annotation_images()





