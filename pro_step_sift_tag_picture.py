# -*- coding:utf-8 -*-
import math

__author__ = 'zmy'

import cv2
# from find_obj import filter_matches,explore_match
import numpy as np
import os
import pydicom


class sift_dicom_jpg:
    def __init__(self, tag_picture):
        self.tag_picture = tag_picture
        self.dicom_total_folder = "H:/medical_data_for_huodong/Images"
        self.patient_id = tag_picture.split("/")[-2]

        self.seq = int(tag_picture.split("/")[-1].split("-")[-1].split(".")[0])

    def get_files(self, path):
        for root, dirs, files in os.walk(path):
            # print(root)  # 当前目录路径
            # print(files)  # 当前路径下所有非目录子文件
            # self.sub_dirs = dirs  # 当前路径下所有子目录
            return files

    def get_dirs(self, path):
        for root, dirs, files in os.walk(path):
            # print(root)  # 当前目录路径
            # print(files)  # 当前路径下所有非目录子文件
            # self.sub_dirs = dirs  # 当前路径下所有子目录
            return dirs

    def find_match_dicom(self):

        years_folder = self.get_dirs(self.dicom_total_folder)

        for year in years_folder:

            dates_folder = self.get_dirs(self.dicom_total_folder + "/" + year)

            for date in dates_folder:
                ids_folder = self.get_dirs(self.dicom_total_folder + "/" + year + "/" + date)

                for id in ids_folder:

                    if id != self.patient_id:
                        continue

                    dicom_list = self.get_files(self.dicom_total_folder + "/" + year + "/" + date + "/" + id)

                    for dicom_name in dicom_list:
                        dcm = pydicom.read_file(
                            self.dicom_total_folder + "/" + year + "/" + date + "/" + id + "/" + dicom_name)

                        if dcm.InstanceNumber == '':
                            continue

                        if int(dcm.InstanceNumber) != self.seq:
                            continue

                        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
                        # DCM.image = DCM.pixel_array

                        img = dcm.image

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

                        cv2.imwrite("img.jpg", temp_img)
                        dicom_with_windows = cv2.imread("img.jpg", cv2.IMREAD_ANYCOLOR)

                        return dicom_with_windows

        return None

    def do_match(self):
        # img1 = cv2.imread("img1.jpg")

        # jpg
        img1 = cv2.imdecode(np.fromfile(self.tag_picture, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)

        # dicom
        img2 = self.find_match_dicom()

        if (img2 is None) or (img1 is None):
            return None

        # jpg_gray
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # dicom
        img2_gray = img2

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        # BFmatcher with default parms
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], None, flags=2)
        cv2.imshow('match', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()

        p1, p2, kp_pairs = self.filter_matches(kp1, kp2, matches, ratio=0.5)

        if p1.shape[0] < 10 or p2.shape[0] < 10:
            print("淘汰：" + self.tag_picture)

            return None

        print(self.tag_picture)

        # self.explore_match('matches', img1_gray, img2_gray, kp_pairs)

        # img3 = cv2.drawMatchesKnn(img1_gray,kp1,img2_gray,kp2,good[:10],flag=2)

        return self.get_right_jpg(img1, img2, kp_pairs)

    def get_right_jpg(self, jpg_img, dicom_img, kp_pairs):
        # temp_mat = jpg_img

        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs])

        number_of_point = p1.shape[0]

        index1 = np.random.randint(0, number_of_point)

        index2 = np.random.randint(0, number_of_point)

        while index2 == index1:
            index2 = np.random.randint(0, number_of_point)

        index3 = np.random.randint(0, number_of_point)

        while index3 == index1 or index3 == index2:
            index3 = np.random.randint(0, number_of_point)

        point1_1 = p1[index1]
        point2_1 = p1[index2]
        point3_1 = p1[index3]

        point1_2 = p2[index1]
        point2_2 = p2[index2]
        point3_2 = p2[index3]

        if point1_1[0] != point1_2[0]:
            slope1 = float(point1_2[1] - point1_1[1]) / (point1_2[0] - point1_1[0])
        else:
            slope1 = 100000000

        if point2_1[0] != point2_2[0]:
            slope2 = float(point2_2[1] - point2_1[1]) / (point2_2[0] - point2_1[0])
        else:
            slope2 = 100000000

        if point3_1[0] != point3_2[0]:
            slope3 = float(point3_2[1] - point3_1[1]) / (point3_2[0] - point3_1[0])
        else:
            slope3 = 100000000

        print(slope1)
        print(slope2)
        print(slope3)

        substract1_2 = np.abs(slope1 - slope2)
        substract1_3 = np.abs(slope1 - slope3)
        substract2_3 = np.abs(slope2 - slope3)

        min = float(np.min(np.array([substract1_2, substract1_3, substract2_3])))

        # point1_1 = p1[index1]
        # point2_1 = p1[index2]
        # point3_1 = p1[index3]
        #
        # point1_2 = p2[index1]
        # point2_2 = p2[index2]
        # point3_2 = p2[index3]

        if np.abs(min - substract1_2) <= 1e-6:
            temp_mat = self.get_four_cut_margin(point1_1, point2_1, point1_2, point2_2, jpg_img, dicom_img)
        elif np.abs(min - substract1_3) <= 1e-6:
            temp_mat = self.get_four_cut_margin(point1_1, point3_1, point1_2, point3_2, jpg_img, dicom_img)
        else:
            temp_mat = self.get_four_cut_margin(point2_1, point3_1, point2_2, point3_2, jpg_img, dicom_img)

        if temp_mat is None:
            return None

        return temp_mat

    def get_four_cut_margin(self, point1_1, point2_1, point1_2, point2_2, jpg_img, dicom_img):

        xd_total = dicom_img.shape[1]
        yd_total = dicom_img.shape[0]

        xj_total = jpg_img.shape[1]
        yj_total = jpg_img.shape[0]

        xd1 = point1_2[0]
        yd1 = point1_2[1]

        xd2 = point2_2[0]
        yd2 = point2_2[1]

        xj1 = point1_1[0]
        yj1 = point1_1[1]

        xj2 = point2_1[0]
        yj2 = point2_1[1]

        cut_up = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_up):
            return None

        yd1 = point1_2[0]
        xd1 = point1_2[1]

        yd2 = point2_2[0]
        xd2 = point2_2[1]

        yj1 = point1_1[0]
        xj1 = point1_1[1]

        yj2 = point2_1[0]
        xj2 = point2_1[1]

        cut_left = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_left):
            return None

        xd1 = xd_total - point1_2[0]
        yd1 = yd_total - point1_2[1]

        xd2 = xd_total - point2_2[0]
        yd2 = yd_total - point2_2[1]

        xj1 = xj_total - point1_1[0]
        yj1 = yj_total - point1_1[1]

        xj2 = xj_total - point2_1[0]
        yj2 = yj_total - point2_1[1]

        cut_right = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_right):
            return None

        yd1 = xd_total - point1_2[0]
        xd1 = yd_total - point1_2[1]

        yd2 = xd_total - point2_2[0]
        xd2 = yd_total - point2_2[1]

        yj1 = xj_total - point1_1[0]
        xj1 = yj_total - point1_1[1]

        yj2 = xj_total - point2_1[0]
        xj2 = yj_total - point2_1[1]

        cut_down = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_down):
            return None

        # if cut_right < 0 or cut_left < 0 or cut_up < 0 or cut_down < 0:
        #     return None

        temp_mat = self.padding_resize(yj_total, xj_total, cut_right, cut_left, cut_up, cut_down, jpg_img)

        try:
            temp_mat = cv2.resize(temp_mat, (yd_total, xd_total))
        except:
            return None

        # cv2.imwrite("cut.jpg", temp_mat)

        return temp_mat

    def padding_resize(self, yj_total, xj_total, cut_right, cut_left, cut_up, cut_down, jpg_img):

        # temp_mat = jpg_img

        # cv2.imshow("jpg", jpg_img)
        # cv2.waitKey()

        if cut_up >= 0 and cut_down >= 0:

            temp_mat = jpg_img[cut_up:(yj_total - cut_down)]
        elif cut_up >= 0 and cut_down < 0:
            temp_mat = jpg_img[cut_up:yj_total]

            padding = np.zeros(shape=[np.abs(cut_down), xj_total, 3])
            temp_mat = np.concatenate((temp_mat, padding), axis=0)

        elif cut_up < 0 and cut_down >= 0:
            temp_mat = jpg_img[0:(yj_total - cut_down)]

            padding = np.zeros(shape=[np.abs(cut_up), xj_total, 3])
            temp_mat = np.concatenate((padding, temp_mat), axis=0)

        else:
            temp_mat = jpg_img
            padding_up = np.zeros(shape=[np.abs(cut_up), xj_total, 3])
            padding_down = np.zeros(shape=[np.abs(cut_down), xj_total, 3])

            temp_mat = np.concatenate((padding_up, temp_mat, padding_down), axis=0)

        yj_total = temp_mat.shape[0]

        # cv2.imshow("jpg", temp_mat)
        # cv2.waitKey()

        if cut_left >= 0 and cut_right >= 0:

            temp_mat = temp_mat[:, cut_left:(xj_total - cut_right)]
        elif cut_left >= 0 and cut_right < 0:
            temp_mat = temp_mat[:, cut_up:xj_total]

            padding = np.zeros(shape=[yj_total, np.abs(cut_right), 3])
            temp_mat = np.concatenate((temp_mat, padding), axis=1)

        elif cut_left < 0 and cut_right >= 0:
            temp_mat = temp_mat[:, 0:(xj_total - cut_right)]

            padding = np.zeros(shape=[yj_total, np.abs(cut_left), 3])
            temp_mat = np.concatenate((padding, temp_mat), axis=1)

        else:
            temp_mat = temp_mat
            padding_up = np.zeros(shape=[yj_total, np.abs(cut_left), 3])
            padding_down = np.zeros(shape=[yj_total, np.abs(cut_right), 3])

            temp_mat = np.concatenate((padding_up, temp_mat, padding_down), axis=1)

        cv2.imwrite("padding.jpg", temp_mat)
        temp_mat = cv2.imread("padding.jpg", cv2.IMREAD_ANYCOLOR)

        # cv2.imshow("padding", temp_mat)
        # cv2.waitKey()
        # print(cut_up)
        # print(cut_down)
        # print(cut_left)
        # print(cut_right)

        return temp_mat

    def compute(self, xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2):

        sum1 = xd2 * (yj1 * xd1 - yd1 * xj1)

        sum2 = xd1 * (yd2 * xj2 - yj2 * xd2)

        sum3 = yd2 * xd1 - yd1 * xd2

        if sum3 == 0:
            print("偏移量计算失败")

            return float('nan')

        # print(str(sum1) + ":" + str(sum2) + ":" + str(sum3))

        return int(float((sum1 + sum2)) / sum3)

    def filter_matches(self, kp1, kp2, matches, ratio=0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = list(zip(mkp1, mkp2))
        return p1, p2, kp_pairs

    def explore_match(self, win, img1, img2, kp_pairs, status=None, H=None):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = np.ones(len(kp_pairs), np.bool)
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

        green = (0, 255, 0)

        index = 0
        for (x1, y1), (x2, y2), inlier in list(zip(p1, p2, status)):

            if index % 20 != 0:
                index = index + 1
                continue
            else:
                index = index + 1

            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)
                cv2.circle(vis, (x1, y1), 5, (0, 0, 255))
                cv2.circle(vis, (x2, y2), 5, (0, 0, 255))

        cv2.imshow(win, vis)
        cv2.imwrite("match.jpg", vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tag_picture = "H:/medical_data_for_huodong/taged_picture/梁邦玉/13355/孙洪君-1-57.jpg"

    sdj = sift_dicom_jpg(tag_picture)
    sdj.do_match()
