import cv2
import numpy as np

from pro_step_circle_profile import profile_processing
from pro_step_sift_tag_picture import sift_dicom_jpg


class recognize_circle:

    def __init__(self, path):
        self.path = path

        self.isLung = True

    # 把原图中的红色提取出来，并转为灰度图
    def get_red(self):

        pp = profile_processing(self.path)

        img = pp.extract_center()
        cv2.imwrite("save_img/extra_big_circle.jpg", img)

        cv2.imshow('extra', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        if img is None:

            sdj = sift_dicom_jpg(self.path)

            img = sdj.do_match()

            if img is None:
                return None, None

            else:
                self.isLung = False

        else:
            self.isLung = True

        res = img
        # cv2.imshow('hehe', res)
        #
        # cv2.waitKey()

        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])

        # 黑背景白圈
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # 黑背景红圈
        res = cv2.bitwise_and(res, res, mask=mask)
        cv2.imwrite("save_img/extra_red_circle.jpg", res)
        cv2.imshow('red', res)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return res, img

    def get_circle(self):

        # 得到只剩红色的图
        red_only_image, img_after_cut = self.get_red()
        print('hahha')
        # 为了生成测试数据，之后注掉
        # if self.isLung == True:
        #     return None
        #
        # if red_only_image is None:
        #     return None

        # print(red_only_image.shape)
        # 灰度化
        gray = cv2.cvtColor(red_only_image, cv2.COLOR_BGR2GRAY)
        # 霍夫变换圆检测
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=1, maxRadius=60)
        # 输出返回值，方便查看类型

        if circles is None:
            print("没有识别到圆")
            return None

        # print(circles)
        # 输出检测到圆的个数
        # print(len(circles[0]))

        print('处理识别到的圆')
        print(circles[0])
        # 根据检测到圆的信息，画出每一个圆
        for circle in circles[0]:
            # 圆的基本信息
            # print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])

            xbool = x > 120 and x < 400
            ybool = y > 60 and y < 430

            if xbool and ybool:
                print(circle)

                id = self.path.split("/")[-2]
                red_only_image = cv2.circle(red_only_image, (x, y), r, (0, 0, 255))
                cv2.imshow('write_red', red_only_image)
                cv2.waitKey()
                cv2.destroyAllWindows()
                # cv2.imwrite("circle_pics/" + id + ".jpg", red_only_image)

                # red_only_image = cv2.circle(red_only_image, (x, y), r, (0, 0, 255))

                # cv2.imshow("red", red_only_image)
                # cv2.waitKey()

                id = self.path.split("/")[-2]

                img_after_cut = img_after_cut[(y - r):(y + r), (x - r):(x + r)]

                # cv2.imwrite("jpg_ori/" + id + ".jpg", img_after_cut)

                return [x, y, r]

            # 半径

            # 在原图用指定颜色标记出圆的位置

        # with open("cut_of_id.txt", "a", encoding="UTF-8") as target:
        #     target.write(self.path.split("/")[-2] + "\n")

        return None


if __name__ == "__main__":
    path = "D:/Mywork/image_coord_regenerate/孙俊杰-1-131.jpg"
    sdj = sift_dicom_jpg(path)
    img = sdj.do_match()
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # rc = recognize_circle(path)
    # circle = rc.get_circle()
