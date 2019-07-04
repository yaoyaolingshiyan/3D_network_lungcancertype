import base_settings
import base_dicom_process
import step1_shift_ct_axis_generate
import glob
import os
import numpy
import math
import cv2
import pandas
from multiprocessing import Pool
import operator

# 图像旋转
def cv_flip(img, cols, rows, degree):
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1.0)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


# 得到重采样后的坐标并保存
# train_coord.txt
def extract_dicom_axis_patient(src_dir):
    # CT 数据集ct提取图片存储目录: D:/Mywork/data/extracted_image/
    problem_list = ['17356', '27783', '28844', '26807', '09065', '12703']
    if src_dir in problem_list:
        return None
    dir_path = base_settings.DICOM_SRC_DIR + src_dir + "/"
    patient_id = src_dir
    slices = base_dicom_process.load_patient(dir_path)
    # print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
    # print("Orientation: ", slices[0].ImageOrientationPatient)
    cos_value = (slices[0].ImageOrientationPatient[0])

    # round 函数是返回浮点数的四舍五入值
    # acos() 函数返回一个数的反余弦值，单位是弧度
    # degrees() 函数将弧度转换为角度
    cos_degree = round(math.degrees(math.acos(cos_value)), 2)
    if cos_degree > 0:
        print(patient_id, ': cos_degree > 0')
        assert False, '坐标有角度偏转'
    pixels = base_dicom_process.get_pixels_hu(slices)
    image = pixels
    img_zxy_shape = image.shape
    # print(img_zxy_shape)

    # true 为没有倒置， false为倒置
    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    # 图像是否倒置
    print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)


    # 得到缩放后的病灶坐标以及肺癌类型（z,x,y,type）,四舍五入取整
    coord_zxyt = step1_shift_ct_axis_generate.get_zxy(patient_id)
    if coord_zxyt:
        for coord_zxy in coord_zxyt:
            coord_zxy[1] = int(pixel_spacing[0]*int(coord_zxy[1]) + 0.5)
            coord_zxy[2] = int(pixel_spacing[1] * int(coord_zxy[2]) + 0.5)
            coord_zxy[0] = int(pixel_spacing[2] * int(coord_zxy[0]) + 0.5)

            if not invert_order:
                # 图像反转
                image = numpy.flipud(image)
                coord_zxy[0] = step1_shift_ct_axis_generate.filter_one_coord(coord_zxy[0], patient_id)
            # print(coord_zxy)
            # 保存用于训练用的（z,x,y）
            with open('D:/Mywork/coord/train_coord.txt', "a", encoding="UTF-8") as target:
                save_str = str(patient_id)+':'+str(coord_zxy[0])+':'\
                           + str(coord_zxy[1])+':'+str(coord_zxy[2])+':'+str(coord_zxy[3]).replace('\r', '')
                target.write(save_str)
        print(patient_id, 'is over!')
    else:
        print(patient_id, 'can not found!')
    return None

# 将txt格式存储的训练坐标存储为csv格式
def extract_train_csv(clean_targetdir_first=False, only_patient_id=None):
    print("Extracting noudle axis and type for train")
    # CT noudle坐标和类型存储目录:D:/Mywork/coord/
    target_dir = base_settings.TRAIN_LABEL
    # 如果target_dir目录下存在文件，则进入if,删除该目录下所有文件
    if clean_targetdir_first and only_patient_id is None:
        print("Cleaning patientID_annotation.csv")
        # 获取target_dir目录下的所有格式文件
        for f_path in glob.glob(target_dir+'*.*'):
            # 删除该文件
            os.remove(f_path)

    if only_patient_id is None:
        # CT 文件夹
        dirs = step1_shift_ct_axis_generate.get_usecoord_patient()
        print('dirs:', dirs)
        if dirs is not None:
            for ct_dir in dirs:
                txt_to_csv(ct_dir)
    else:
        txt_to_csv(only_patient_id)

# 遍历
def extract_dicom_axis(clean_targetdir_first=False, only_patient_id=None):
    print("Extracting noudle axis and type for train")
    # CT noudle坐标和类型存储目录:D:/Mywork/coord/
    target_dir = base_settings.TRAIN_COORD
    # 如果target_dir目录下存在文件，则进入if,删除该目录下所有文件
    if clean_targetdir_first and only_patient_id is None:
        print("Cleaning train_coord.txt")
        # 获取target_dir目录下的所有格式文件
        if os.path.exists(target_dir + "train_coord.txt"):
            # 删除该文件
            os.remove(target_dir + "train_coord.txt")

    if only_patient_id is None:
        # CT 文件夹
        dirs = step1_shift_ct_axis_generate.get_usecoord_patient()
        print('dirs count is : ', len(dirs))
        if dirs is not None:
            for ct_dir in dirs:
                extract_dicom_axis_patient(ct_dir)
    else:
        extract_dicom_axis_patient(only_patient_id)

# src_dir是CT文件夹中的病人id文件夹(只能是文件夹)(eg:'3255')
# 将ct图重采样后存储为png以及mask
def extract_dicom_images_patient(src_dir):
    problem_list = ['17356', '27783', '28844', '26807', '09065', '12703']
    if src_dir in problem_list:
        return None
    print('src_dir is: ', src_dir)
    # CT 数据集ct提取图片存储目录: D:/Mywork/data/extracted_image/
    # target_dir = settings.DICOM_EXTRACT_DIR
    target_dir = 'D:/Mywork/data/extracted_image/'
    ct_dirs = os.listdir(target_dir)
    if src_dir in ct_dirs:
        print(src_dir, ' had saved!')
    else:
        dir_path = base_settings.DICOM_SRC_DIR + src_dir + "/"
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
        # dicom_basic_process.look_HU(image)
        # print('first_image.shape is: ', image.shape)

        # true 为没有倒置， false为倒置
        invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
        # 图像是否倒置
        # print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])

        pixel_spacing = slices[0].PixelSpacing
        pixel_spacing.append(slices[0].SliceThickness)

        image = base_dicom_process.rescale_patient_images(image, pixel_spacing, base_settings.TARGET_VOXEL_MM,
                                                          verbose=False)
        if image is None:
            print('the CT is to big , can not resample!')
            return None
        # print('rescale_image.shape:', image.shape)

        if not invert_order:
            # 图像反转
            image = numpy.flipud(image)

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
            cv2.imwrite(img_path, org_img * 255)
            cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)
        print(patient_id, 'is over!')

# 遍历CT
def extract_dicom_images(clean_targetdir_first=False, only_patient_id=None):
    print("Extracting CT images to png and mask")
    # CT数据集ct提取图片存储目录:D:/Mywork/data/extracted_image/
    target_dir = 'D:/Mywork/data/extracted_image/'
    # 如果target_dir目录下存在文件，则进入if,删除该目录下所有文件
    # if clean_targetdir_first and only_patient_id is None:
    #     print("Cleaning target dir")
    #     # 获取target_dir目录下的所有格式文件
    #     for file_path in glob.glob(target_dir + "*.*"):
    #         # 删除该文件
    #         os.remove(file_path)

    if only_patient_id is None:
        # CT 文件夹
        dirs = base_dicom_process.get_new_right_axis_patient()
        if dirs is not None:
            # 创建进程池
            pool = Pool(2)
            # 第一个参数是函数，第二个参数是一个迭代器，将迭代器中的数字作为参数依次传入函数中
            pool.map(extract_dicom_images_patient, dirs)
            # print('dirs:', dirs)
            # print(len(dirs))
            # for ctdir in dirs:
            #     if ctdir == '13957' or ctdir == '17356':
            #         continue
            #     extract_dicom_images_patient(ctdir)
    else:
        extract_dicom_images_patient(only_patient_id)

# 将坐标存储为csv文件
def txt_to_csv(patientID):

    coord_dir = 'D:/Mywork/coord/train_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')
    all_lines = []
    print('first patientID is:', patientID)
    print('patientID length is:', len(patientID))
    while line:
        patient = line.split(':')[0].replace(' ', '')
        print('patient:', patient, ':', len(patient))
        if patientID == patient:
            coord_z = line.split(':')[1]
            coord_x = line.split(':')[2]
            coord_y = line.split(':')[-2]
            lung_type = line.split(':')[-1]
            print('patientID:', patientID)
            all_lines.append([patientID, lung_type, coord_x, coord_y, coord_z])
            print(patientID+':'+lung_type)
            line = f.readline().decode('UTF-8')
        else:
            line = f.readline().decode('UTF-8')
    f.close()
    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z"])
    df_annos.to_csv(base_settings.TRAIN_LABEL + patientID + "_annotations.csv", index=False, encoding='UTF-8')

if __name__ == '__main__':

    # 1、CT图重采样并保存png和mask
    extract_dicom_images(clean_targetdir_first=False, only_patient_id='25391')
    # 2、noudle坐标及类型重采样成训练使用的坐标
    # extract_dicom_axis(clean_targetdir_first=False, only_patient_id=None)
    # 将txt格式存储的训练坐标存储为csv格式
    # extract_train_csv(clean_targetdir_first=True, only_patient_id=None)