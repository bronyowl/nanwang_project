# 主要功能：加载预训练模型，对测试图像进行超分辨率处理，并可视化结果。

import sys
sys.path.append('./utils_paddle')
from utils_paddle.paddle_aux import *
import paddle
import os
from utils import *
from models import SRResNet
import time
from PIL import Image
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

def generate_test_images(file_paths, output_folder, time_range=(0, 10000)):
    """
    为多个场景生成测试图像
    :param file_paths: 包含多个场景数据文件路径的列表
    :param output_folder: 输出文件夹
    :param time_range: 时间范围元组 (start, end)
    """
    all_Vm = []
    all_MeasurePMU = []
    all_MeasureRTU = []
    img_count = 0
    
    for file_path in file_paths:
        # 读取电压数据
        Vm = pd.read_excel(file_path, header=None).values  # shape: (110000, 39)
        
        # 生成PMU和RTU量测
        MeasurePMU_Vm = np.zeros_like(Vm)
        measureRTU_Vm = np.zeros_like(Vm)
        
        # 添加不同误差的测量噪声
        error = [0.001, 0.003]  # PMU和RTU的误差
        for o in range(Vm.shape[1]):  # 遍历节点
            MeasurePMU_Vm[:, o] = Vm[:, o] * (1 + norm.rvs(size=Vm.shape[0], scale=error[0]))
            measureRTU_Vm[:, o] = Vm[:, o] * (1 + norm.rvs(size=Vm.shape[0], scale=error[1]))
        
        # RTU数据降采样处理（每100个时间点取一个值）
        for node in range(Vm.shape[1]):  # 对每个节点
            for t in range(0, Vm.shape[0], 100):  # 每100个时间点
                if t + 100 <= Vm.shape[0]:
                    measureRTU_Vm[t:t+100, node] = measureRTU_Vm[t, node]
        
        all_Vm.append(Vm)
        all_MeasurePMU.append(MeasurePMU_Vm)
        all_MeasureRTU.append(measureRTU_Vm)
        
        # 为每个节点生成测试图像
        scaler = MinMaxScaler(feature_range=(0, 255))
        for i in range(39):  # 39个节点
            # 获取RTU数据并降采样（每100个点取一个）
            data = measureRTU_Vm[time_range[0]:time_range[1], i][::100]  # 从10000降到100个点
            matrix = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            n = 10  # 低分辨率图像尺寸
            normalized_matrix = matrix.reshape(n, n)  # 10x10矩阵
            RGB_matrix = normalized_matrix.astype(np.uint8)
            rgb_image = np.dstack((RGB_matrix, RGB_matrix, RGB_matrix))
            
            image_path = os.path.join(output_folder, f'{img_count}.bmp')
            Image.fromarray(rgb_image).save(image_path)
            img_count += 1
    
    return np.concatenate(all_Vm), np.concatenate(all_MeasurePMU), np.concatenate(all_MeasureRTU)

if __name__ == '__main__':
    # 创建测试图像输出目录
    output_folder = './datasets/test/images'
    os.makedirs(output_folder, exist_ok=True)
    
    # 合并处理rd和dl场景
    file_paths = [
        './datasets/train/rd/Vm.xlsx',
        './datasets/train/dl/Vm.xlsx'
    ]
    
    # 生成测试图像和量测数据
    Vm, MeasurePMU_Vm, measureRTU_Vm = generate_test_images(
        file_paths,
        output_folder,
        time_range=(0, 10000)
    )
    
    # 选择要测试的节点编号（与原代码保持一致，使用第12个节点）
    test_node = 12
    imgPath = os.path.join('./datasets/test/images', f'{test_node-1}.bmp')  # 因为生成图像时索引从0开始
    
    # 创建结果目录
    os.makedirs('./results/srresnet', exist_ok=True)
    
    # 添加模型配置参数
    large_kernel_size = 9
    small_kernel_size = 3
    n_channels = 64
    n_blocks = 5
    scaling_factors = 8
    scaling_factor = 10
    
    # 加载模型
    srresnet_checkpoint = './checkpoints/srresnet/srresnet.pdparams'
    checkpoint = paddle.load(path=srresnet_checkpoint)
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
                       small_kernel_size=small_kernel_size,
                       n_channels=n_channels,
                       n_blocks=n_blocks,
                       scaling_factors=scaling_factors)
    srresnet = srresnet.to(device)
    srresnet.set_state_dict(state_dict=checkpoint['model'])
    srresnet.eval()
    model = srresnet
    # 处理图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
    
    # 保存和评估结果
    Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
    Bicubic_img.save('./results/srresnet/test_bicubic.bmp')
    
    # 模型推理和结果可视化
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(axis=0)
    start = time.time()
    lr_img = lr_img.to(device)
    
    with paddle.no_grad():
        sr_img = model(lr_img).squeeze(axis=0).cpu().detach()
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('./results/srresnet/test_srresnet.bmp')
    
    print('用时  {:.3f} 秒'.format(time.time() - start))
    
    # 结果评估和绘图
    image = Image.open('./results/srresnet/test_srresnet.bmp').convert('L')
    image_vector = np.array(image).flatten() / 255.0
    
    # 使用正确的时间范围和降采样
    time_points = 100  # RTU采样点数
    x = np.linspace(0, 100, time_points)  # 100秒，100个点
    
    # 获取对应节点的真实值和RTU测量值（降采样后）
    Vm_i = Vm[0:10000:100, test_node-1]  # 每100个点取一个
    RTU_i = measureRTU_Vm[0:10000:100, test_node-1]
    
    plt.figure()
    plt.plot(x, image_vector[:time_points], '-r', label='SR Result')
    plt.plot(x, Vm_i, '-k', label='True value')
    plt.plot(x, RTU_i, '--b', label='RTU measurement')
    plt.xlabel('time/s')
    plt.ylabel('Voltage amplitude/p.u.')
    plt.legend()
    plt.show()
