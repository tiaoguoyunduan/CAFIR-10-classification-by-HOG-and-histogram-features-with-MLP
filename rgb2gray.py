import numpy as np

def rgb2gray(rgb):
    """
    将RGB图片转换成灰度图片：公式img = 0.299 * R + 0.587 * G + 0.144 * B

      输入:
        rgb : RGB 图片
      返回:
        gray : 灰度图片

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
