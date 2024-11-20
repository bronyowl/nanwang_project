# 主要功能：提供各种辅助函数，包含图像转换、数据增强、学习率调整等多种辅助函数。

import paddle
import os
from PIL import Image
import json
import random
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')
rgb_weights = paddle.to_tensor(data=[65.481, 128.553, 24.966], dtype='float32'
    ).to(device)
imagenet_mean = paddle.to_tensor(data=[0.485, 0.456, 0.406], dtype='float32'
    ).unsqueeze(axis=1).unsqueeze(axis=2)
imagenet_std = paddle.to_tensor(data=[0.229, 0.224, 0.225], dtype='float32'
    ).unsqueeze(axis=1).unsqueeze(axis=2)
imagenet_mean_cuda = paddle.to_tensor(data=[0.485, 0.456, 0.406], dtype=
    'float32').to(device).unsqueeze(axis=0).unsqueeze(axis=2).unsqueeze(axis=3)
imagenet_std_cuda = paddle.to_tensor(data=[0.229, 0.224, 0.225], dtype=
    'float32').to(device).unsqueeze(axis=0).unsqueeze(axis=2).unsqueeze(axis=3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    创建训练集和测试集列表文件.
        参数 train_folders: 训练文件夹集合; 各文件夹中的图像将被合并到一个图片列表文件里面
        参数 test_folders: 测试文件夹集合; 每个文件夹将形成一个图片列表文件
        参数 min_size: 图像宽、高的最小容忍值
        参数 output_folder: 最终生成的文件列表,json格式
    """
    print('\n正在创建文件列表... 请耐心等待.\n')
    train_images = list()
    for d in train_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print('训练集中共有 %d 张图像\n' % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)
    for d in test_folders:
        test_images = list()
        test_name = d.split('/')[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print('在测试集 %s 中共有 %d 张图像\n' % (test_name, len(test_images)))
        with open(os.path.join(output_folder, test_name +
            '_test_images.json'), 'w') as j:
            json.dump(test_images, j)
    print('生成完毕。训练集和测试集文件列表已保存在 %s 下\n' % output_folder)


def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, '无法转换图像源格式 %s!' % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]',
        'imagenet-norm', 'y-channel'}, '无法转换图像目标格式t %s!' % target
    if source == 'pil':
        img = paddle.vision.transforms.functional.to_tensor(img)
    elif source == '[0, 1]':
        pass
    elif source == '[-1, 1]':
        img = (img + 1.0) / 2.0
    if target == 'pil':
        img = Image.fromarray(np.uint8(img.numpy() * 255).transpose(1, 2, 0)).convert('RGB')
    elif target == '[0, 255]':
        img = 255.0 * img
    elif target == '[0, 1]':
        pass
    elif target == '[-1, 1]':
        img = 2.0 * img - 1.0
    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda
    elif target == 'y-channel':
        img = paddle.matmul(x=255.0 * img.transpose(perm=[0, 2, 3, 1])[:, 4
            :-4, 4:-4, :], y=rgb_weights) / 255.0 + 16.0
    return img


class ImageTransforms(object):
    """
    图像变换.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
        hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """
        if self.split == 'train':
            left = random.randint(0, img.width - self.crop_size)
            top = random.randint(0, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
            int(hr_img.height / self.scaling_factor)), Image.BICUBIC)
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
        return lr_img, hr_img


class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    丢弃梯度防止计算过程中梯度爆炸.

    :参数 optimizer: 优化器，其梯度将被截断
    :参数 grad_clip: 截断值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    保存训练结果.

    :参数 state: 逐项预保存内容
    """
    paddle.save(obj=state, path=filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率.

    :参数 optimizer: 需要调整的优化器
    :参数 shrink_factor: 调整因子，范围在 (0, 1) 之间，用于乘上原学习率.
    """
    print('\n调整学习率.')
    for param_group in optimizer.param_groups:
        param_group['lr'] *= shrink_factor


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        paddle.nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        paddle.nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != 1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clip_(
            min=-0.025, max=0.025)
        paddle.nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(tuple(Img.shape)[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :],
            data_range=data_range)
    return PSNR / tuple(Img.shape)[0]


def data_augmentation(image, mode):
    x = np
    perm_0 = list(range(x.ndim))
    perm_0[image] = 1, 2, 0
    perm_0[1, 2, 0] = image
    out = x.transpose(perm=perm_0)
    if mode == 0:
        out = out
    elif mode == 1:
        out = np.flipud(out)
    elif mode == 2:
        out = np.rot90(k=out)
    elif mode == 3:
        out = np.rot90(k=out)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(k=2)
    elif mode == 5:
        out = np.rot90(k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(k=3)
    elif mode == 7:
        out = np.rot90(k=3)
        out = np.flipud(out)
    x = np
    perm_1 = list(range(x.ndim))
    perm_1[out] = 2, 0, 1
    perm_1[2, 0, 1] = out
    return x.transpose(perm=perm_1)

def make_grid(tensors, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    if not isinstance(tensors, paddle.Tensor):
        tensors = paddle.to_tensor(tensors)

    if normalize:
        if range is not None:
            mn, mx = range
        else:
            mn, mx = tensors.min(), tensors.max()
        tensors = (tensors - mn) / (mx - mn)

    if scale_each:
        for t in tensors:
            mn, mx = t.min(), t.max()
            t.subtract_(mn).divide_(mx - mn)

    if tensors.ndim == 2:
        tensors = tensors.unsqueeze(0)
    if tensors.ndim == 3:
        tensors = tensors.unsqueeze(0)

    nmaps = tensors.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensors.shape[2] + padding), int(tensors.shape[3] + padding)
    grid = paddle.full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding : (y + 1) * height, x * width + padding : (x + 1) * width] = tensors[k]
            k += 1
    return grid