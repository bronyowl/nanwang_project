# 主要功能：读取电压数据，进行归一化处理，然后将数据转换为图像格式并保存，创建训练图像列表。

import sys
sys.path.append('D/研一/南网数研院/GAN/torch2padle/utils_paddle')
from utils_paddle.paddle_aux import *
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from PIL import Image
BUS = [4, 8, 11, 15, 18, 27, 31]
file_path = 'C:\\Users\\cbbis\\Desktop\\Data_fusion\\x10\\train_7\\V.xlsx'
Vm = pd.read_excel(file_path, header=None).values
scaler = MinMaxScaler(feature_range=(0, 255))
for i in range(64, 71):
    data = Vm[90000:100000, BUS[i - 64] - 1].reshape(-1, 1)
    matrix = scaler.fit_transform(data)
    n = 100
    normalized_matrix = matrix.reshape(-1, n)
    RGB_matrix = normalized_matrix.astype(np.uint8)
    rgb_image = np.dstack((RGB_matrix, RGB_matrix, RGB_matrix))
    image_path = os.path.join(
        'C:\\Users\\cbbis\\Desktop\\Data_fusion\\x10\\train_real', f'{i}.bmp')
    Image.fromarray(rgb_image).save(image_path)
