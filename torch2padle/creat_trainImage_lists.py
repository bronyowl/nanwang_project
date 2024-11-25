# 主要功能：读取电压数据，进行归一化处理，然后将数据转换为图像格式并保存，创建训练图像列表。

import sys
sys.path.append('./utils_paddle')
from utils_paddle.paddle_aux import *
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from PIL import Image
from scipy.stats import norm

# 读取rd和dl场景的数据
rd_file_path = './datasets/train/rd/Vm.xlsx'
dl_file_path = './datasets/train/dl/Vm.xlsx'

# PMU噪声参数
error = 0.001  # PMU噪声标准差

# 修改BUS节点选择
BUS = [q for q in range(1, 40)]  # 1到39的所有节点

def generate_pmu_images(file_path, output_folder, start_idx):
    # 读取电压数据
    Vm = pd.read_excel(file_path, header=None).values
    
    # 生成PMU量测
    MeasurePMU_Vm = np.zeros_like(Vm)
    for o in range(Vm.shape[1]):
        MeasurePMU_Vm[:, o] = Vm[:, o] * (1 + norm.rvs(size=Vm.shape[0], scale=error))
    
    scaler = MinMaxScaler(feature_range=(0, 255))
    
    # 修改循环范围以适应39个节点
    for i in range(len(BUS)):
        # 使用10000个点来生成100x100的图像
        data = MeasurePMU_Vm[10000:20000, BUS[i]-1].reshape(-1, 1)
        matrix = scaler.fit_transform(data)
        n = 100  # 设置分辨率为100x100
        normalized_matrix = matrix.reshape(n, n)
        RGB_matrix = normalized_matrix.astype(np.uint8)
        rgb_image = np.dstack((RGB_matrix, RGB_matrix, RGB_matrix))
        
        # 修改图像保存的编号方式
        image_path = os.path.join(output_folder, f'{start_idx + i}.bmp')
        Image.fromarray(rgb_image).save(image_path)

if __name__ == '__main__':
    output_folder = './datasets/train/images'
    os.makedirs(output_folder, exist_ok=True)
    
    # 为rd和dl场景分别生成39张图像
    # 假设从1开始编号
    generate_pmu_images(rd_file_path, output_folder, start_idx=1)  # 生成1-39号图像
    generate_pmu_images(dl_file_path, output_folder, start_idx=40)  # 生成40-78号图像
