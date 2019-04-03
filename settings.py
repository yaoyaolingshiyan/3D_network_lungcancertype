import os
import datetime

COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

# 工作进程数
WORKER_POOL_SIZE = 4
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

# 各数据地址
BASE_DIR = 'D:/Mywork/'
DICOM_SRC_DIR = 'D:/Mywork/data/src_dicom/'
DICOM_EXTRACT_DIR = 'D:/Mywork/data/extracted_image/'
TRAIN_COORD = 'D:/Mywork/coord/'
TRAIN_LABEL = 'D:/Mywork/data/generated_trainlabel/'
TRAIN_DATA = 'D:/Mywork/data/generated_traindata/'
PREDICT_CUBE = 'D:/Mywork/data/predict_cube/'

# 计时器
class Stopwatch(object):

    def start(self):
        self.start_time = Stopwatch.get_time()

    def get_elapsed_time(self):
        current_time = Stopwatch.get_time()
        res = current_time - self.start_time
        return res

    def get_elapsed_seconds(self):
        elapsed_time = self.get_elapsed_time()
        res = elapsed_time.total_seconds()
        return res

    @staticmethod
    def get_time():
        res = datetime.datetime.now()
        return res

    @staticmethod
    def start_new():
        res = Stopwatch()
        res.start()
        return res


