import cv2
import numpy as np


# jpg标记图像中，ct有黑边，而dicom没有黑边，需要将中间的圆提取出来然后resize到512
class profile_processing:

    def __init__(self, pic_path):
        self.pic_path = pic_path

    # def find_out_circle(self):
    #     red_only_image = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    #
    #     if red_only_image.shape[0] != 512 or red_only_image.shape[1] != 512:
    #         red_only_image = cv2.resize(red_only_image, (512, 512), interpolation=cv2.INTER_CUBIC)
    #
    #     # print(red_only_image.shape)
    #     # 灰度化
    #     gray = cv2.cvtColor(red_only_image, cv2.COLOR_BGR2GRAY)
    #
    #     # 霍夫变换圆检测
    #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=150,
    #                                maxRadius=250)
    #     # 输出返回值，方便查看类型
    #
    #     if circles is None:
    #         print("没有识别到圆")
    #         return None
    #
    #     # print(circles)
    #     # 输出检测到圆的个数
    #     # print(len(circles[0]))
    #
    #     print('处理识别到的圆')
    #     # 根据检测到圆的信息，画出每一个圆
    #     for circle in circles[0]:
    #         # 圆的基本信息
    #         # print(circle[2])
    #         # 坐标行列
    #         x = int(circle[0])
    #         y = int(circle[1])
    #         r = int(circle[2])
    #
    #         xbool = x > 200 and x < 300
    #         ybool = y > 200 and y < 300
    #
    #         if xbool and ybool:
    #             print(circle)
    #
    #             red_only_image = cv2.circle(red_only_image, (x, y), r, (0, 0, 255))
    #
    #     # cv2.imshow("hehe", red_only_image)
    #     # cv2.waitKey()

    def extract_center(self):

        ori = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)

        if ori.shape[0] != 512 or ori.shape[1] != 512:
            ori = cv2.resize(ori, (512, 512), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        cv2.imshow('mygray', gray)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # 霍夫变换圆检测
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=150,
                                   maxRadius=250)
        for circle in circles[0]:
            # 圆的基本信息
            # print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])

            xbool = x > 200 and x < 300
            ybool = y > 200 and y < 300

            if xbool and ybool:

                if x < r or y < r:
                    continue

                x_start = x - r
                x_end = x + r

                y_start = y - r

                y_end = y + r

                red_only_image = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)

                red_only_image = red_only_image[y_start:y_end, x_start:x_end]
                red_only_image = cv2.resize(red_only_image, (512, 512), interpolation=cv2.INTER_CUBIC)

                return red_only_image

        return None


if __name__ == "__main__":
    pic_path = "D:/Mywork/医学图像标记代码/李奎权-1-106.jpg"

    pp = profile_processing(pic_path)
    # pp.find_out_circle()
    pp.extract_center()
