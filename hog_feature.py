import rgb2gray
import numpy as np
from scipy import ndimage

def rgb2gray(rgb):
    """
    将RGB图片转换成灰度图片：公式img = 0.299 * R + 0.587 * G + 0.144 * B

      输入:
        rgb : RGB 图片
      返回:
        gray : 灰度图片

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def hog_feature(im):
    """
    计算图片的梯度方向直方图（HOG）特征
         从 skimage.feature.hog 中修改而来
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : 灰度图片或者RBG图片

      Returns:
        feat: HOG 特征

    """

    # 如果图像维数是3维，则转换成灰度图
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # 图片尺寸
    orientations = 9  # 梯度直方图的数量
    cx, cy = (8, 8)  # 一个单元的像素个数

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = ndimage.uniform_filter(temp_mag, size=(cx, cy))[int(cx / 2)::cx, int(cy / 2)::cy].T

    return orientation_histogram.ravel()

    
if __name__ == '__main__':
    from data_utils import load_CIFAR10
    #加载Cifar10数据集，并输出数据集的维数
    cifar10_dir = 'D:\DIP_project\cifar-10-python\cifar-10-batches-py'
    X_train,y_train,X_test,y_test = load_CIFAR10(cifar10_dir) # Y是标签（0-9）
    print(X_train.shape) # (10000, 32, 32, 3)

    train_num = 50000 # set train data num here
    hsv_feather_Xtrain = np.ones( (50000,144) )
    for i in range(0,train_num): # range是小于9的
        test_im = X_train[i]
        test_hsv = hog_feature(test_im)
        # print(test_hsv)
        hsv_feather_Xtrain[i] = test_hsv

    np.savetxt("hsv_feather_Xtrain.txt",hsv_feather_Xtrain) #缺省按照'%.18e'格式保存数据，以空格分隔
    load_hsv_Xtrain = np.loadtxt("hsv_feather_Xtrain.txt")
    print(load_hsv_Xtrain)
    print('------------------------------------')
    print(hsv_feather_Xtrain)

    hsv_feather_Xtest = np.ones( (10000,144) )
    test_num = 10000 # set teat data num here
    for j in range(0,test_num): # range是小于9的
        test_im = X_test[j]
        test_hsv = hog_feature(test_im)
        # print(test_hsv)
        hsv_feather_Xtest[j] = test_hsv

    np.savetxt("hsv_feather_Xtest.txt",hsv_feather_Xtest) #缺省按照'%.18e'格式保存数据，以空格分隔
    load_hsv_Xtest = np.loadtxt("hsv_feather_Xtest.txt")
    print('------------------------------------')
    print(load_hsv_Xtest)
    print('------------------------------------')
    print(hsv_feather_Xtest)