import numpy as np
import scipy.ndimage
import os
import base_settings
import base_dicom_process
import math
import cv2
from multiprocessing import Pool
import glob

# 检验train_coord.txt文件有哪几个ID还没有被重采样
def inspect_dicom_to_resample():
    without_list = []
    txt_id_list = get_origin_coord_withoutrepeat('train_coord.txt')
    resample_img_path = 'D:/Mywork/data/extracted_image/'
    resample_list = os.listdir(resample_img_path)
    for pat in txt_id_list:
        if pat not in resample_list:
            without_list.append(pat)
    print(without_list)


# 得到无重复的有坐标提取信息的病人id
def get_origin_coord_withoutrepeat(txt_path):
    src_dir = 'D:/Mywork/image_coord_regenerate/new_coord/'
    src_dir = src_dir+txt_path
    id_list = []
    f = open(src_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patient_id = str(line.split(':')[0])

        if patient_id not in id_list:
            id_list.append(patient_id)
        line = f.readline().decode('UTF-8')
    f.close()
    return id_list


# 图像旋转
def cv_flip(img, cols, rows, degree):
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1.0)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

# 原重采样方法之后坐标有偏差，这里使用新的采样方法
# 输入image坐标顺序为zyx
def resample(image, spacing, new_spacing=1.):
    print(spacing)
    resize_factor_z = spacing[2] / new_spacing
    resize_factor_x = spacing[1] / new_spacing
    resize_factor_y = spacing[0] / new_spacing
    new_real_shape_z = np.round(image.shape[0] * resize_factor_z)
    new_real_shape_x = np.round(image.shape[2] * resize_factor_x)
    new_real_shape_y = np.round(image.shape[1] * resize_factor_y)

    real_resize_factor_z = new_real_shape_z / image.shape[0]
    real_resize_factor_x = new_real_shape_x / image.shape[2]
    real_resize_factor_y = new_real_shape_y / image.shape[1]

    real_resize_factor = [real_resize_factor_z, real_resize_factor_y, real_resize_factor_x]

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def add_ct_window(img):
    y = img.shape[0]
    x = img.shape[1]
    temp_img = np.zeros(shape=[y, x])

    row = 0

    while row < y:
        col = 0
        while col < x:
            if int(img[row][col]) <= -160:
                temp_img[row][col] = int(0)
            elif int(img[row][col]) >= 240:
                temp_img[row][col] = int(255)
            else:
                temp_img[row][col] = int((float(img[row][col]) + 160) * 255 / 400)

            col = col + 1

        row = row + 1

    return temp_img

def extract_dicom_images(clean_targetdir_first=False, only_patient_id=None):
    print("Extracting CT images")
    # CT数据集ct提取图片存储目录:D:/Mywork/data/extracted_image/
    target_dir = 'D:/Mywork/image_coord_regenerate/extract_img/'
    # 如果target_dir目录下存在文件，则进入if,删除该目录下所有文件
    if clean_targetdir_first and only_patient_id is None:
        print("Cleaning target dir")
        # 获取target_dir目录下的所有格式文件
        for file_path in glob.glob(target_dir + "*.*"):
            # 删除该文件
            os.remove(file_path)

    if only_patient_id is None:
        # CT 文件夹
        dirs = get_origin_coord_withoutrepeat('use_coord.txt')
        if dirs is not None:
            # 创建进程池
            pool = Pool(2)
            # 第一个参数是函数，第二个参数是一个迭代器，将迭代器中的数字作为参数依次传入函数中
            pool.map(extract_dicom_images_patient, dirs)

    else:
        extract_dicom_images_patient(only_patient_id)

# src_dir是CT文件夹中的病人id文件夹(只能是文件夹)(eg:'3255')
# 将ct图重采样后存储为png以及mask
def extract_dicom_images_patient(src_dir):
    print('src_dir is: ', src_dir)
    # CT 数据集ct提取图片存储目录: D:/Mywork/data/extracted_image/
    target_dir = 'D:/Mywork/data/extract_img/'
    ct_dirs = os.listdir(target_dir)
    if src_dir in ct_dirs:
        print(src_dir, ' had saved!')
    else:
        dir_path = 'D:/Mywork/data/src_dicom/' + src_dir + "/"
        patient_id = src_dir
        slices = base_dicom_process.load_patient(dir_path)
        # print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
        # print("Orientation: ", slices[0].ImageOrientationPatient)
        # assert slices[0].ImageOrientationPatient == [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
        cos_value = (slices[0].ImageOrientationPatient[0])

        # round 函数是返回浮点数的四舍五入值
        # acos() 函数返回一个数的反余弦值，单位是弧度
        # degrees() 函数将弧度转换为角度
        cos_degree = round(math.degrees(math.acos(cos_value)), 2)
        if cos_degree > 0:
            print(patient_id, ': cos_degree > 0')
        pixels = base_dicom_process.get_pixels_hu(slices)
        image = pixels
        # print(image.shape)
        # dicom_basic_process.look_HU(image)
        # print('first_image.shape is: ', image.shape)

        # true 为没有倒置， false为倒置
        invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
        # 图像是否倒置
        # print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])

        pixel_spacing = slices[0].PixelSpacing
        pixel_spacing.append(slices[0].SliceThickness)
        print(pixel_spacing)
        image = resample(image, pixel_spacing, new_spacing=1)
        print(image.shape)
        if image is None:
            print('the CT is to big , can not resample!')
            return None
        # print('rescale_image.shape:', image.shape)

        if not invert_order:
            # 图像反转
            image = np.flipud(image)

        for i in range(image.shape[0]):  # image:(873,500,500)
            # CT 转化为图片的存放地址
            patient_dir = target_dir + patient_id + "/"
            # print(i, ':', patient_dir)
            if not os.path.exists(patient_dir):
                os.mkdir(patient_dir)
            # 这里 rjust() 返回一个原字符串右对齐,并使用0填充至长度 4 的新字符串(如果指定的长度小于字符串的长度则返回原字符串)
            img_path = patient_dir + "img_" + str(i).rjust(4, '0') + "_i.png"
            org_img = image[i]
            # print(org_img)
            # if there exists slope,rotation image with corresponding degree
            if cos_degree > 0.0:
                org_img = cv_flip(org_img, org_img.shape[1], org_img.shape[0], cos_degree)

            img, mask = base_dicom_process.get_segmented_lungs(org_img.copy())
            # print('*********************************')
            # print(img)
            # print(operator.eq(org_img, img))
            org_img = base_dicom_process.normalize_hu(org_img)
            # org_img = add_ct_window(org_img)
            cv2.imwrite(img_path, org_img * 255)
            # cv2.imwrite(img_path, org_img)
            cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)
        print(patient_id, 'is over!')

if __name__ == '__main__':
    print('Hello, zmy')
    # extract_dicom_images(clean_targetdir_first=True, only_patient_id=None)
    extract_dicom_images_patient('3630')
    # 检查重采样文件是否缺失
    # inspect_dicom_to_resample()
