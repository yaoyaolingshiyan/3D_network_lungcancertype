import settings
import numpy
import pydicom
import os
import cv2
import glob
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi


# 得到CT扫描切片厚度，加入到切片信息中
def load_patient(src_dir):
    # src_dir是病人CT文件夹地址
    slices = [pydicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# 提取CT图像素值（-4000，4000），CT图的像素值是由HU值表示的
def get_pixels_hu(slices):
    image = numpy.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16)
    # should be possible as values should always be low enough(<32k)
    image = image.astype(numpy.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024,so air is approximately 0
    # CT扫描边界之外的灰度值固定为-2000(dicom和mhd都是这个值)。第一步是设定这些值为0，当前对应为空气（值为0）
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(numpy.float64)
            image[slice_number] = image[slice_number].astype(numpy.int16)
        image[slice_number] += numpy.int16(intercept)

    return numpy.array(image, dtype=numpy.int16)

# 将输入图像的像素值（-1024，2000）归一化到0-1之间
def normalize_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

# 查看一下病人扫描的HU值分布情况
def look_HU(pixels):
    patient_pixels = pixels
    plt.hist(patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel('HU')
    plt.ylabel('Frequence')
    plt.show()

# 图像重采样
def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm, is_mask_image=False, verbose=False):
    if verbose:
        print("Spacing: ", org_spacing_xyz)
        print("init_images_zyx_Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    # 插值方法
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    # opencv assumes y, x, channels umpyarray, so y = z, 这里按照resize_y 扩大 Z
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    # swapaxes函数是交换维度
    # 交换z和y维，变成yxz维度，res:(512,512,873)
    res = res.swapaxes(0, 2)
    # 交换y和x维，变成xyz维度，res：(512,512,873)
    res = res.swapaxes(0, 1)
    # print("Shape: ", res.shape)
    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)
    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        # 变成zyx维度顺序（873，512，512）
        # 因为res的z轴维度（CT图像数）多于512张，cv2无法一次处理，所以分部分处理
        res = res.swapaxes(0, 2)
        # print('res.shape:', res.shape)
        res1 = res[:512]
        # print('res1.shape:', res1.shape)
        res2 = res[512:]
        # print('res2.shape:', res2.shape)
        # xyz维度顺序(512,512,512)
        res1 = res1.swapaxes(0, 2)
        # xyz维度顺序(512,512,361)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        # print('res1_resized_shape:', res1.shape)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        # print('res2_resized_shape:', res2.shape)
        # zyx维度顺序：res1:(512,500,500), res2:(361,500,500)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        # print('res_vstack_shape:', res.shape)
        # xyz维度，res:(500,500,873)
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    # zyx维度，res：(873,500,500)
    res = res.swapaxes(0, 2)
    # zxy维度，res：(873,500,500)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res


# 该函数分割出CT切片里肺部组织
def get_segmented_lungs(im, plot=True):

    # Step 1: Convert into a binary image.
    binary = im < -400
    # print('binary.shape:', binary.shape)
    # print(binary)

    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # print('clear.shape:', cleared.shape)
    # print(cleared)
    # Step 3: Label the image.
    label_image = label(cleared)
    # print('label.shape:', label_image.shape)
    # print(label_image)
    # cv2.imshow('label_image', label_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    # Step 6: Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall.
    selem = disk(10)  # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)

    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    return im, binary

# 从 extracted_image 读取图像并将一个病人的图像合在一起，例如：349张330*330的图像
# 返回一个（349，330，330）的三维矩阵（z,y,x）
def load_patient_images(patient_id, base_dir=None, wildcard="*.*", exclude_wildcards=[]):
    if base_dir == None:
        base_dir = settings.DICOM_EXTRACT_DIR
    # D:/Mywork/data/extracted_image/patient_id/
    src_dir = base_dir + patient_id + "/"

    # 排除没有的文件夹
    if not os.path.exists(src_dir):
        return None

    # D:/Mywork/data/extracted_image/patient_id/*_i.png
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1, ) + im.shape) for im in images]
    res = numpy.vstack(images)
    return res


# 每个图片是由多个小图片连在一起的，这个函数把小图像分割出来存进列表并返回
def load_cube_img(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert rows * size == img.shape[0]
    assert cols * size == img.shape[1]
    res = numpy.zeros((rows * cols, size, size))

    img_height = size   # 48
    img_width = size    # 48

    for row in range(rows):    # 6
        for col in range(cols):  # 8
            src_y = row * img_height
            src_x = col * img_width
            # res[0] = img[0:48,0:48], res[1] = img[0:48, 48:96], res[7] = [0:48, 336:384]
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]

    return res
