import os
import shutil
import numpy

# 将某个目录下的CT文件转存到一起
def save_ct():
    src_dir = 'I:/2016/'
    dst_dir = 'D:/Mywork/data/src_dicom/'
    dirpath = os.listdir(src_dir)
    print(dirpath)
    for dir in dirpath:
        year_dir = src_dir+dir+'/'
        # print(year_dir)
        names = os.listdir(year_dir)
        # print(names)
        for name in names:
            name_dir = year_dir+name+'/'
            print(name_dir)
            dst = dst_dir + name + '/'
            if not os.path.exists(dst):
                os.makedirs(dst)
            for dirpath, dirnames, filenames in os.walk(name_dir):
                for filename in filenames:
                    file_dir = name_dir+filename
                    size = os.path.getsize(file_dir)
                    print(size)
                    if size > 500000:
                        file_dst = dst + filename
                        shutil.copyfile(file_dir, file_dst)
            print(name, ' is saved!')

# 从原坐标文件里筛选出需要的坐标转存成新文件
# 原z坐标反了，这里产生新的z坐标=ct数-原z坐标
# 针对批量坐标转换:改变Z坐标为 len（ct）-z
# 该函数使用new_right_axis.txt
# use_coord.txt
def filter_coord_one():
    ct_src_dir = 'D:/Mywork/data/src_dicom/'
    extract_img_dir = 'D:/Mywork/data/extracted_image/'
    src_dir = 'D:/Mywork/coord/new_right_axis.txt'
    ct_list = os.listdir(extract_img_dir)
    print(ct_list)
    print(len(ct_list))
    f = open(src_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patientid = str(line.split(':')[0])
        if patientid in ct_list:
            print(patientid)
            number_ct = os.listdir(ct_src_dir+patientid+'/')
            print('number of ct :', len(number_ct))
            lung_type = str(line.split(':')[1].split('-')[1])
            coord_z = int(line.split(':')[1].split('-')[-1].split('.')[0])
            coord_x = int(line.split(':')[-1].split('-')[0])
            coord_y = int(line.split(':')[-1].split('-')[1])
            new_coord_z = len(number_ct)-coord_z
            new_line = patientid + ':' + str(new_coord_z) + ':' + str(coord_x) + ':' + str(coord_y)+':'+lung_type
            with open('D:/Mywork/coord/use_coord.txt', "a", encoding="UTF-8") as target:
                target.write(new_line+'\n')
        line = f.readline().decode('UTF-8')

    f.close()

# 弃用，right_axis.txt没有记录肺癌类型
# 该函数使用right_axis.txt
def filter_coord_two():
    ct_src_dir = 'D:/Mywork/data/src_dicom/'
    src_dir = 'D:/Mywork/coord/right_axis.txt'
    ct_list = os.listdir(ct_src_dir)
    # print(ct_list)
    f = open(src_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patientID = str(line.split('-')[0])
        if patientID in ct_list:
            print(patientID)
            number_ct = os.listdir(ct_src_dir+patientID+'/')
            print('number of ct :', len(number_ct))
            lung_type = str(line.split('-')[-4])
            coord_z = int(line.split('-')[-4])
            coord_x = int(line.split('-')[-3])
            coord_y = int(line.split('-')[-2])
            new_coord_z = len(number_ct)-coord_z
            line = patientID+':'+str(new_coord_z)+':'+str(coord_x)+':'+str(coord_y)
            with open('D:/Mywork/coord/use_coord.txt', "a", encoding="UTF-8") as target:
                target.write(line+'\n')
        line = f.readline().decode('UTF-8')

    f.close()

# 针对单个坐标转换:改变Z坐标为 len（ct）-z
def filter_one_coord(axis_z, patientID):
    ct_src_dir = 'D:/Mywork/data/src_dicom/'
    # ct_list = os.listdir(ct_src_dir)
    number_ct = os.listdir(ct_src_dir + patientID + '/')
    # print('number of ct :', len(number_ct))
    new_coord_z = len(number_ct) - axis_z
    return new_coord_z



# 提取指定文件坐标（z,x,y, lung_type）:可能存在多个
def get_zxy(patientID):
    coord_dir = 'D:/Mywork/coord/transer_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')

    coord_zxyt = []
    while line:
        patient = str(line.split(':')[0])
        if patientID == patient:
            coord_z = str(line.split(':')[1])
            coord_x = str(line.split(':')[2])
            coord_y = str(line.split(':')[-2])
            lung_type = str(line.split(':')[-1])
            coord_zxyt.append([coord_z, coord_x, coord_y, lung_type])
        line = f.readline().decode('UTF-8')
    f.close()
    return coord_zxyt

# 得到无重复的有坐标提取信息的病人id
def get_usecoord_patient():
    src_dir = 'D:/Mywork/coord/use_coord.txt'
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

# transer_coord.txt
def transform_type(id_list):
    type_list = ['1', '2', '3', '4']
    m_list = []
    l_list = []
    for idd in id_list:
        coord_zxyt = get_zxy(idd)
        print(coord_zxyt)
        for co_zxyt in coord_zxyt:
            str1 = co_zxyt[-1].replace('\n', '').replace('\r', '')
            print(str1)
            if str1 in type_list:
                m_list.append(idd)
                l_list.append(str1)
                break

    coord_dir = 'D:/Mywork/coord/use_coord.txt'
    f = open(coord_dir, 'rb')
    line = f.readline().decode('UTF-8')
    while line:
        patient = str(line.split(':')[0])
        num = m_list.index(patient)
        line = line.replace(str(line.split(':')[-1]), str(l_list[num]))
        with open('D:/Mywork/coord/transer_coord.txt', "a", encoding="UTF-8") as target:
            target.write(line + '\n')
        line = f.readline().decode('UTF-8')
    f.close()

    return None


if __name__ == '__main__':
    # 将CT文件以病人ID为文件夹命名存储到一起
    save_ct()
    # 针对批量坐标转换:改变Z坐标为 len（ct）-z : use_coord.txt
    # filter_coord_one()
    # 3、归一化肺癌类型，将'淋巴结'，'骨转移'等标签归类到0-5: transer_coord.txt
    # idlist = get_usecoord_patient()
    # print(idlist)
    # print(len(idlist))
    # transform_type(idlist)




