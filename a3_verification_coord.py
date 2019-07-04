# 该文件验证提取出的坐标和重采样后的坐标对应的图像位置是否一样
import cv2
import pydicom
import numpy
import os
import glob
import pandas
import shutil
import base_dicom_process


# 读取每个文件CT数目，CT数多余300张的文件，记录id
def get_ct_length():
    ct_path = 'I:/src_dicom/'
    ct_list = os.listdir(ct_path)
    print(ct_list)
    print(len(ct_list))
    for patient in ct_list:
        patient_path = ct_path+patient
        patient_list = os.listdir(patient_path)
        num = len(patient_list)
        if num > 350:
            with open('D:/Mywork/image_coord_regenerate/ct_out_range.txt', "a", encoding="UTF-8") as target:
                save_str = patient+':'+str(num)+'\n'
                target.write(save_str)
        print(len(patient_list))

# 验证：得到我们记录的CT的Z坐标，在sort后，实际位置（Z）
# 验证我们转换后的z坐标和它的真实坐标是否一样
def get_ct_older():
    dicom_path = 'I:/src_dicom/27799/'
    slices = load_patient(dicom_path)
    # print(slices[0].pixel_array)
    print(slices[0].SOPInstanceUID)
    # for i in range(0, len(slices)):
    #     print(slices[i].SOPInstanceUID)
    #     number = str(slices[i].SOPInstanceUID).split('.')[-1]
    #     if number == '27':
    #         print(str(i) + ': ' + number)


# 提取指定文件坐标（x,y,z,lung_type）:可能存在多个
def get_xyz(patientID):
    coord_dir = 'D:/Mywork/image_coord_regenerate/new_coord/use_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    coord_zxyt = []
    while line:
        patient = str(line.split(':')[0])
        if patientID == patient:
            coord_z = str(line.split(':')[1])
            coord_x = str(line.split(':')[2])
            coord_y = str(line.split(':')[3])
            lung_type = str(line.split(':')[4])
            noudle_r = int(str(line.split(':')[5]).replace('\r', '').replace('\n', ''))
            if lung_type in ['1', '2', '3', '4', '5']:
                coord_zxyt.append([coord_x, coord_y, coord_z, lung_type, noudle_r])
        line = f.readline().decode('UTF-8')
    f.close()
    return coord_zxyt

# 读取use_coord.txt文件信息
def get_use_coord_list():
    all_lines = []
    coord_dir = 'D:/Mywork/image_coord_regenerate/new_coord/use_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    while line:
        patient = str(line.split(':')[0])
        coord_z = str(line.split(':')[1])
        coord_x = str(line.split(':')[2])
        coord_y = str(line.split(':')[3])
        lung_type = str(line.split(':')[4])
        lung_r = int(line.split(':')[5].replace('\r', '').replace('\n', ''))
        all_lines.append([patient, coord_x, coord_y, coord_z, lung_type, str(lung_r)])
        # print(patient)
        line = f.readline().decode('UTF-8')

    f.close()
    return all_lines


def verif_coord(name_list):
    # src_dir = 'D:/Mywork/data/extracted_image/'
    src_dir = 'D:/Mywork/image_coord_regenerate/extract_img/'
    for i in range(0, len(name_list)):
        patientID = name_list[i][0]
        coord_x = int(name_list[i][1])
        coord_y = int(name_list[i][2])
        coord_z = name_list[i][3]
        coord_r = int(float(name_list[i][5]))
        search_dir = patientID+'/' + "img_" + str(coord_z).rjust(4, '0') + "_i.png"
        img = cv2.imread(src_dir+search_dir)

        if img is not None:
            img = cv2.circle(img, (coord_x, coord_y), coord_r, (0, 0, 255))
            cv2.imshow(search_dir, img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            print(i, ':', patientID+' over!')
        else:
            print(search_dir+' 没有')
            continue

# 得到CT扫描切片厚度，加入到切片信息中
def load_patient(src_dir):
    # src_dir是病人CT文件夹地址
    slices = [pydicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# 遍历CT目录
def traverse_dicom():
    ct_list = []
    year_list = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    for year in year_list:
        month_list = os.listdir('I:/'+year+'/')
        for month in month_list:
            patient_list = os.listdir('I:/'+year+'/'+month+'/')
            for patient in patient_list:
                # print(patient)
                ct_list.append(str('I:/'+year+'/'+month+'/'+patient+'/'))
    print(ct_list)
    print(len(ct_list))
    return ct_list

# 将use_coord.txt中的绝对坐标存储为相对坐标
def coord_absolute_to_relative():
    coord_list = []
    x_size = 512.
    y_size = 512.
    id_list = get_origin_coord_withoutrepeat('use_coord.txt')
    for ctid in id_list:
        z_size = float(len(os.listdir('F:/src_dicom/'+ctid+'/')))
        id_info = get_xyz(ctid)
        for id_in in id_info:
            rela_x = int(id_in[0])/x_size
            rela_y = int(id_in[1])/y_size
            rela_z = int(id_in[2])/z_size
            lung_type = id_in[3]
            rela_r = float(id_in[4])/x_size
            info_str = ctid+':'+str(rela_x)+':'+str(rela_y)+':'+str(rela_z)+':'+str(rela_r)+':'+lung_type
            coord_list.append(info_str)
            print(info_str)
    print(len(coord_list))


# 记录CT种类
def get_CT_series_description(patient_path):
    # print(patient_path)
    series_list = []
    slices = [pydicom.read_file(patient_path + '/' + s) for s in os.listdir(patient_path)]
    for slice in slices:
        str_name = str(slice.SeriesDescription).replace(' ', '')
        if str_name not in series_list:
            series_list.append(str_name)
    patientid = patient_path.split('/')[-2]
    serise_quantity = len(series_list)
    series_list.append(patient_path)
    series_list.append(patientid)
    series_list.append(str(serise_quantity))
    # with open('D:/Mywork/ct_description.txt', "a", encoding="UTF-8") as target:
    #     save_str = str('***'.join(series_list))
    #     target.write(save_str + '\n')

    serise1 = 'WBStandard'
    serise2 = 'CTAttenCorHeadIn3.75thk'
    if serise1 not in series_list and serise2 not in series_list:
        with open('D:/Mywork/without_two_maindescription.txt', "a", encoding="UTF-8") as target:
            save_str2 = str('***'.join(series_list))
            target.write(save_str2 + '\n')

    print(patientid, ':', series_list, ':', serise_quantity)

def bianli_shift():
    dicom_path = traverse_dicom()
    txt_id_list = get_origin_coord_withoutrepeat('new_right_axis.txt')
    print(txt_id_list)
    print(len(txt_id_list))
    # shift_CT('I:/2008/20080504/3107/')
    for order, dpath in enumerate(dicom_path):
        print(order, ':', dpath)
        patientid = dpath.split('/')[-2]
        if patientid in txt_id_list:
            shift_CT(dpath)

# 转存CT
def shift_CT(dicom_path):
    patientid = dicom_path.split('/')[-2]
    target_path = 'F:/src_dicom/'
    dicom_path_list = os.listdir(dicom_path)
    if not os.path.exists(target_path+patientid):
        os.mkdir(target_path+patientid)
    for dicom in dicom_path_list:
        slice = pydicom.read_file(dicom_path+dicom)
        str_name = str(slice.SeriesDescription)
        if str_name == 'CT Atten Cor Head In 3.75thk':
            shutil.copyfile(dicom_path+dicom, target_path+patientid+'/'+dicom)
        elif str_name == 'CT Atten Cor Head In 3.75 thk':
            shutil.copyfile(dicom_path+dicom, target_path+patientid+'/'+dicom)
        elif str_name == 'WB Standard':
            shutil.copyfile(dicom_path + dicom, target_path + patientid + '/' + dicom)
        elif str_name == 'REC':
            shutil.copyfile(dicom_path + dicom, target_path + patientid + '/' + dicom)
        else:
            continue

def read_CT_type():
    all_lines = []
    coord_dir = 'D:/Mywork/ct_description.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    while line:
        all_lines.append(str(line).replace('\r', '').replace('\n', '').split('***'))
        # print(patient)
        line = f.readline().decode('UTF-8')

    f.close()
    return all_lines

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

# 验证提取的Z坐标和CT是否对应: 提取Z坐标对应的CT，存储为png格式
def verify_coord_ct():
    ct_list = get_use_coord_list()
    print(ct_list)
    print(len(ct_list))

    for ct_inf in ct_list:
        print(ct_inf)
        all_list = os.listdir('F:/src_dicom/')
        if ct_inf[0] in all_list:
            dicom_path = 'F:/src_dicom/' + ct_inf[0] + '/'
            slices = load_patient(dicom_path)
            pixels = base_dicom_process.get_pixels_hu(slices)
            numbe = int(ct_inf[3])
            img = base_dicom_process.normalize_hu(pixels[numbe])
            img_path = 'D:/Mywork/image_coord_regenerate/verify_img/'+ct_inf[0]+'_'+ct_inf[3]+'.png'
            cv2.imwrite(img_path, img * 255)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        else:
            print('do not find ', ct_inf[0])

        print(ct_inf[0], 'is over')

# 验证提取的Z坐标和CT是否对应: 提取某一张Z坐标对应的CT，存储为png格式
def one_verify_coord_ct():
    dicom_path = 'F:/src_dicom/21703/'
    slices = load_patient(dicom_path)
    pixels = base_dicom_process.get_pixels_hu(slices)
    img = base_dicom_process.normalize_hu(pixels[121])
    img_path = 'D:/Mywork/image_coord_regenerate/verify_img/21703_121.png'
    cv2.imwrite(img_path, img * 255)


# 验证最初提取的坐标是否正确:在提取图上绘制病灶圆
def verify_png_coord():
    use_coord_list = get_use_coord_list()
    img_save_path = 'D:/Mywork/image_coord_regenerate/verify_img/'
    imgs_list = os.listdir(img_save_path)
    print(imgs_list)
    print(len(imgs_list))
    for img in imgs_list:
        patientid = img.split('_')[0]
        patient_z = img.split('_')[1].split('.')[0]
        for use_coord in use_coord_list:
            if str(use_coord[0])==patientid and use_coord[3]==patient_z:
                need_img = cv2.imread(img_save_path+img, cv2.IMREAD_GRAYSCALE)
                need_img = cv2.circle(need_img, (int(use_coord[1]), int(use_coord[2])), int(use_coord[5]), (0, 0, 255))
                cv2.imshow(img, need_img)
                cv2.waitKey()
                cv2.destroyAllWindows()


# 验证最初提取的坐标是否正确:提取标记jpg
def extract_tag_jpg():
    ct_list = get_use_coord_list()
    print(ct_list)
    print(len(ct_list))
    tag_path = 'I:/graduation_design/taged_picture/allmingzitechang/'
    all_list = os.listdir(tag_path)
    tag_img_list = []
    for ct_inf in ct_list:
        print(ct_inf)
        if ct_inf[0] in all_list:
            jpg_list = os.listdir(tag_path+ct_inf[0]+'/')
            if 'Thumbs.db' in jpg_list:
                jpg_list.remove('Thumbs.db')
            for jpg_name in jpg_list:
                print(jpg_name)
                info_list = str(jpg_name).split('-')
                patient_type = str(info_list[1])
                if patient_type == 'SCLC':
                    patient_type = '5'
                if len(info_list)==3 and patient_type in ['1', '2', '3', '4', '5']:
                    patient_z = jpg_name.split('-')[2].split('.')[0]
                else:
                    continue
                if patient_type==ct_inf[4] and patient_z==ct_inf[3]:
                    print('hahaha')
                    tag_img_list.append(patient_type)
                    img_path = 'D:/Mywork/image_coord_regenerate/tag_img/'+ct_inf[0]+'_'+ct_inf[3]+'.jpg'
                    shutil.copyfile(tag_path+ct_inf[0]+'/'+jpg_name, img_path)

        else:
            print('do not find ', ct_inf[0])

        print(ct_inf[0], 'is over')

    print('tag img list is', len(tag_img_list))

# 对照查看tag图片和依据z坐标提取的ct图片
def look_tag_ct():
    tag_path = 'D:/Mywork/image_coord_regenerate/tag_img/'
    ct_path = 'D:/Mywork/image_coord_regenerate/verify_img/'
    jpgs_list = os.listdir(tag_path)
    pngs_list = os.listdir(ct_path)
    for png in pngs_list:
        find_jpg = False
        png_older = png.split('.')[0]
        for jpg in jpgs_list:
            jpg_older = jpg.split('.')[0]
            # if str(jpg_older+'.png') not in pngs_list:
            #     print('tag_img:', jpg, 'without verify!')
            #     break
            if jpg_older == png_older:
                find_jpg = True
                png1 = cv2.imread(ct_path + png, cv2.IMREAD_GRAYSCALE)
                jpg1 = cv2.imread(tag_path + jpg, cv2.IMREAD_ANYCOLOR)
                # print(jpg)
                # cv2.imshow('verify_png', png1)
                # cv2.imshow('tag_jpg', jpg1)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                break

        if not find_jpg:
            print(png, 'not find!')

# 对照查看tag图片和提取的ct图片的（X,Y）坐标是否准确
def look_tag_ct_xy():
    tag_path = 'D:/Mywork/image_coord_regenerate/tag_img/'
    ct_path = 'D:/Mywork/image_coord_regenerate/verify_img/'
    use_info_list = get_use_coord_list()
    jpgs_list = os.listdir(tag_path)
    for use_info in use_info_list:
        # print(use_info[0], ':', use_info[3])
        for older, jpg in enumerate(jpgs_list):
            jpg_id = jpg.split('.')[0].split('_')[0]
            jpg_z = jpg.split('.')[0].split('_')[1]
            # print(jpg_id, ':', jpg_z)
            if jpg_id==use_info[0] and jpg_z==use_info[3]:
                print(older, ':', jpg)
                img_name = jpg_id+'_'+jpg_z
                img_png = cv2.imread(ct_path+img_name+'.png', cv2.IMREAD_ANYCOLOR)
                img_jpg = cv2.imread(tag_path+jpg, cv2.IMREAD_ANYCOLOR)
                img_png = cv2.circle(img_png, (int(use_info[1]), int(use_info[2])), int(use_info[5]), (0, 0, 255))
                cv2.imshow('png:'+jpg_id, img_png)
                cv2.imshow('jpg:'+jpg_id, img_jpg)
                cv2.waitKey()
                cv2.destroyAllWindows()

# 查看CT转换图片大小，
def look_png_size():
    png_path = 'D:/Mywork/image_coord_regenerate/verify_img/'
    pngs = glob.glob(png_path+'*.png')
    print(len(pngs))
    for png in pngs:
        img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if img.shape !=(512,512):
            print(png)
        # print(img.shape)

def count_type_number():
    one_list = []
    two_list = []
    three_list = []
    four_list = []
    five_list = []
    info_list = get_use_coord_list()
    print(len(info_list))
    for info in info_list:
        if info[4] == '1':
            one_list.append(info)
        elif info[4] == '2':
            two_list.append(info)
        elif info[4] == '3':
            three_list.append(info)
        elif info[4] == '4':
            four_list.append(info)
        else:
            five_list.append(info)
    print(len(one_list), ':', len(two_list), ':', len(three_list), ':', len(four_list), ':', len(five_list))
if __name__ == '__main__':
    print('Hello, zmy')

    # 验证重采样后的图像病灶坐标是否正确
    # lines_list = get_name_list()
    # print(lines_list)
    # print(len(lines_list))
    # verif_coord(lines_list[450:])
    # get_ct_older()
    # get_ct_length()


    # 得到CT种类
    # dicom_path = traverse_dicom()
    # for order, dpath in enumerate(dicom_path):
    #     print(order, ':', dpath)
    #     get_CT_series_description(dpath)

    # 转存CT
    # 转换单个
    # shift_CT('I:/2015/20150107/21790/')
    bianli_shift()

    # 计算转存的CT数量
    # path1 = 'F:/src_dicom/'
    # id_list = os.listdir(path1)
    # for i in id_list:
    #     number = os.listdir(path1+i+'/')
    #     print(i, ':', len(number))

    # verify_coord_ct()

    # verify_png_coord()

    # extract_tag_jpg()
    # look_tag_ct()
    # one_verify_coord_ct()

    # ct_list = get_origin_coord_withoutrepeat('use_coord.txt')
    # for ctid in ct_list:
    #     patient_list = get_xyz(ctid)
    #     if len(patient_list)>1:
    #         print(ctid, ':', len(patient_list))

    # my = get_xyz('26611')
    # print(my)

    # look_png_size()

    # look_tag_ct_xy()

    # coord_absolute_to_relative()

    # count_type_number()

