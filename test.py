import cv2
import numpy as np
import glob
import os

ix = -1
iy = -1
cx = -1
cy = -1
r = 0
drawing = False

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


def draw_circle(event, x, y, flags, param):
    global ix, iy, cx, cy, r, drawing
    if event==cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        drawing = True
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing:
            cx = x
            cy = y
            r = int(abs(cx-ix))
    elif event==cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), r, (0, 0, 255), 1)


# def show_img(img_path):
#

if __name__ == '__main__':

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
            if jpg_id == use_info[0] and jpg_z == use_info[3]:
                # print(older, ':', jpg)
                # 显示tag图片
                img_jpg = cv2.imread(tag_path + jpg, cv2.IMREAD_ANYCOLOR)
                cv2.imshow('jpg:' + jpg_id, img_jpg)


                img_name = jpg_id + '_' + jpg_z
                img = cv2.imread(ct_path + img_name + '.png', cv2.IMREAD_ANYCOLOR)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.namedWindow('image')

                while (1):
                    cv2.setMouseCallback('image', draw_circle)
                    cv2.imshow('image', img)
                    keycode = cv2.waitKey(1)

                    # 刷新这张图像，重新标记病灶
                    if keycode == ord('a'):
                        img = cv2.imread(ct_path + img_name + '.png', cv2.IMREAD_ANYCOLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        cv2.destroyWindow('image')
                        continue
                    # 保存病灶坐标和半径
                    elif keycode == ord('d'):
                        print(use_info[0], str(cx), str(cy), str(r))
                        line = use_info[0] + ':' + use_info[3] + ':' + str(cx) + ':' + str(cy) + ':' + str(r)+':'+use_info[-2]
                        with open('D:/Mywork/image_coord_regenerate/new_coord/sign_label.txt', "a",
                                  encoding="UTF-8") as target:
                            target.write(line + '\n')
                        break
                    # 跳过这张图像，记录这张图像编号
                    elif keycode == ord('s'):
                        line = use_info[0] + ':' + use_info[3]+':bad'
                        print(line)
                        with open('D:/Mywork/image_coord_regenerate/new_coord/bad_id.txt', "a",
                                  encoding="UTF-8") as target:
                            target.write(line + '\n')
                        break
                cv2.destroyAllWindows()

