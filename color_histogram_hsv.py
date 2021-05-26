import numpy as np
import matplotlib

# 直方图柱子的数量是可以改的。
def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    计算HSV颜色特征
    输入：
    - im : H x W x C 的RGB数组
    - nbin : 直方图柱状的数量
    - xmin : 最小像素值（缺省值：0）
    - xmax : 最大像素值（缺省值：255）
    - normalized : 是否归一化（缺省值：True）
    返回：
    - imhist : 图像的颜色直方图

    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist

if __name__ == '__main__':
    from data_utils import load_CIFAR10
    #加载Cifar10数据集，并输出数据集的维数
    cifar10_dir = 'D:\DIP_project\cifar-10-python\cifar-10-batches-py'
    X_train,y_train,X_test,y_test = load_CIFAR10(cifar10_dir)
    print(X_train.shape) # (10000, 32, 32, 3)

    histogram_feather_Xtrain = np.ones( (50000,10) ) # 这里的10是nbin
    train_num = 50000 # set train data num here
    for i in range(0,train_num): # range是小于9的
        test_im = X_train[i]
        test_hsv = color_histogram_hsv(test_im)
        # print(test_hsv)
        histogram_feather_Xtrain[i] = test_hsv

    np.savetxt("histogram_feather_Xtrain.txt",histogram_feather_Xtrain) #缺省按照'%.18e'格式保存数据，以空格分隔
    load_histogram_Xtrain = np.loadtxt("histogram_feather_Xtrain.txt")
    print(load_histogram_Xtrain)
    print('------------------------------------')
    print(histogram_feather_Xtrain)

    histogram_feather_Xtest = np.ones( (10000,10) )
    test_num = 10000  # set test data num here
    for j in range(0,test_num): # range是小于9的
        test_im = X_test[j]
        test_hsv = color_histogram_hsv(test_im)
        # print(test_hsv)
        histogram_feather_Xtest[j] = test_hsv

    np.savetxt("histogram_feather_Xtest.txt",histogram_feather_Xtest) #缺省按照'%.18e'格式保存数据，以空格分隔
    load_histogram_Xtest = np.loadtxt("histogram_feather_Xtest.txt")
    print('------------------------------------')
    print(load_histogram_Xtest)
    print('------------------------------------')
    print(histogram_feather_Xtest)


    # test_im = X_train[0]
    # test_hsv = color_histogram_hsv(test_im)
    # print(test_hsv)
