# 主要功能：加载预训练模型，对测试图像进行超分辨率处理，并可视化结果。

import sys
sys.path.append('D:\\研一\\南网数研院\\GAN\torch2padle/utils_paddle')
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
MeasurePMU_Vm = np.zeros((39, 10000))
measureRTU_Vm = np.zeros((39, 10000))
file_path = 'C:\\Users\\cbbis\\Desktop\\Data_fusion\\x10\\train_7\\V.xlsx'
Vm = pd.read_excel(file_path, header=None).values
for o in range(10000):
    error = [0.001, 0.003]
    MeasurePMU_Vm[:, o] = Vm[:, o] * (1 + norm.rvs(size=39, scale=error[0]))
    measureRTU_Vm[:, o] = Vm[:, o] * (1 + norm.rvs(size=39, scale=error[1]))
for k in range(1, 101):
    MeasureRTU_Vm[:, k - 1] = measureRTU_Vm[:, (k - 1) * 100]
scaler = MinMaxScaler(feature_range=(0, 255))
n = 10
i = 12
data = MeasureRTU_Vm[i - 1, :]
matrix = scaler.fit_transform(data.reshape(-1, 1)).flatten()
normalized_matrix = matrix.reshape(-1, n)
RGB_matrix = normalized_matrix.astype(np.uint8)
rgb_image = np.dstack((RGB_matrix, RGB_matrix, RGB_matrix))
image_path = os.path.join(
    'C:\\Users\\cbbis\\Desktop\\Data_fusion\\x10\\test_6.8_7.1', f'{i}.bmp')
Image.fromarray(rgb_image).save(image_path)
imgPath = 'C:/Users/cbbis/Desktop/Data_fusion/x10/test_6.8_7.1/i.bmp'
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 5
scaling_factor = 10
scaling_factors = 8
target_size = 100, 100
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')
if __name__ == '__main__':
    srresnet_checkpoint = (
        'C:/Users/cbbis/Desktop/Data_fusion/x10/results_shuff3_5/checkpoint_srresnet.pth'
        )
    checkpoint = paddle.load(path=srresnet_checkpoint)
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
        small_kernel_size=small_kernel_size, n_channels=n_channels,
        n_blocks=n_blocks, scaling_factors=scaling_factors)
    srresnet = srresnet.to(device)
    srresnet.set_state_dict(state_dict=checkpoint['model'])
    srresnet.eval()
    model = srresnet
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
    Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.
        height * scaling_factor)), Image.BICUBIC)
    Bicubic_img.save(
        'C:/Users/cbbis/Desktop/Data_fusion/x10/results_shuff3_5/test_bicubic.bmp'
        )
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(axis=0)
    start = time.time()
    lr_img = lr_img.to(device)
    with paddle.no_grad():
        sr_img = model(lr_img).squeeze(axis=0).cpu().detach()
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save(
            'C:/Users/cbbis/Desktop/Data_fusion/x10/results_shuff3_5/test_srresnet.bmp'
            )
    print('用时  {:.3f} 秒'.format(time.time() - start))
image_path = (
    'C:/Users/cbbis/Desktop/Data_fusion/x10/results_shuff3_5/test_srresnet.bmp'
    )
image = Image.open(image_path).convert('L')
image_vector = np.array(image).flatten() / 255.0
A = MeasurePMU_Vm[i, :].T
Vm_i = Vm[i, :]
Original_Value = image_vector * (A.max() - A.min()) + A.min()
x = np.arange(0.01, 100.01, 0.01)
plt.figure()
plt.plot(x, Original_Value[:10000], '-r', label='RTU pseudo-measurement')
plt.plot(x, Vm_i[:10000], '-k', label='True value')
plt.hold(True)
plt.xlabel('time/s')
plt.ylabel('Voltage amplitude/p.u.')
plt.legend()
plt.show()
