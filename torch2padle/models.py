import sys
sys.path.append('D:\\研一\\南网数研院\\GAN\torch2padle/utils_paddle')
from utils_paddle.paddle_aux import *
import paddle
import math


class ConvolutionalBlock(paddle.nn.Layer):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()
        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}
        layers = list()
        layers.append(paddle.nn.Conv2D(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=kernel_size // 2))
        if batch_norm is True:
            layers.append(paddle.nn.BatchNorm2D(num_features=out_channels))
        if activation == 'prelu':
            layers.append(paddle.nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(paddle.nn.LeakyReLU(negative_slope=0.2))
        elif activation == 'tanh':
            layers.append(paddle.nn.Tanh())
        self.conv_block = paddle.nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)
        return output


class SubPixelConvolutionalBlock(paddle.nn.Layer):
    """
    子像素卷积模块, 包含卷积, 像素清洗和激活层.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :参数 kernel_size: 卷积核大小
        :参数 n_channels: 输入和输出通道数
        :参数 scaling_factor: 放大比例
        """
        super(SubPixelConvolutionalBlock, self).__init__()
        self.conv = paddle.nn.Conv2D(in_channels=n_channels, out_channels=
            n_channels * scaling_factor ** 2, kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.pixel_shuffle = paddle.nn.PixelShuffle(upscale_factor=
            scaling_factor)
        self.prelu = paddle.nn.PReLU()

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)
        return output


class ResidualBlock(paddle.nn.Layer):
    """
    残差模块, 包含两个卷积模块和一个跳连.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :参数 kernel_size: 核大小
        :参数 n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）
        """
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels,
            out_channels=n_channels, kernel_size=kernel_size, batch_norm=
            True, activation='PReLu')
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels,
            out_channels=n_channels, kernel_size=kernel_size, batch_norm=
            True, activation=None)

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)
        """
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output + residual
        return output


class SRResNet(paddle.nn.Layer):
    """
    SRResNet模型
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels
        =64, n_blocks=16, scaling_factors=8, target_size=(50, 50),
        dropout_rate=0.5):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(SRResNet, self).__init__()
        scaling_factors = int(scaling_factors)
        assert scaling_factors in {2, 4, 8}, '放大比例必须为 2、 4 或 8!'
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=
            n_channels, kernel_size=large_kernel_size, batch_norm=False,
            activation='PReLu')
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.residual_blocks = paddle.nn.Sequential(*[ResidualBlock(
            kernel_size=small_kernel_size, n_channels=n_channels) for i in
            range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels,
            out_channels=n_channels, kernel_size=small_kernel_size,
            batch_norm=True, activation=None)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        n_subpixel_convolution_blocks = int(math.log2(scaling_factors))
        self.subpixel_convolutional_blocks = paddle.nn.Sequential(*[
            SubPixelConvolutionalBlock(kernel_size=small_kernel_size,
            n_channels=n_channels, scaling_factor=2) for i in range(
            n_subpixel_convolution_blocks)])
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels,
            out_channels=3, kernel_size=large_kernel_size, batch_norm=False,
            activation='Tanh')
        self.target_size = target_size

    def forward(self, lr_imgs):
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)
        output = self.dropout(output)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = self.dropout(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        output = paddle.nn.functional.interpolate(x=output, size=self.
            target_size, mode='bilinear', align_corners=False)
        sr_imgs = self.conv_block3(output)
        return sr_imgs


class Generator(paddle.nn.Layer):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, 
                 n_blocks=16, scaling_factors=8, target_size=(100, 100)):  # 添加target_size参数
        super(Generator, self).__init__()
        self.net = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size, 
            n_channels=n_channels,
            n_blocks=n_blocks, 
            scaling_factors=scaling_factors,
            target_size=target_size  # 传递给SRResNet
        )
    
    def forward(self, lr_imgs):
        sr_imgs = self.net(lr_imgs)
        return sr_imgs
        
    def set_net_state_dict(self, state_dict):
        self.net.set_state_dict(state_dict)


class Discriminator(paddle.nn.Layer):
    """
    SRGAN判别器
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=
        1024, dropout_rate=0.5):
        """
        参数 kernel_size: 所有卷积层的核大小
        参数 n_channels: 初始卷积层输出通道数, 后面每隔一个卷积层通道数翻倍
        参数 n_blocks: 卷积块数量
        参数 fc_size: 全连接层连接数
        """
        super(Discriminator, self).__init__()
        in_channels = 3
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2
                ) if i % 2 is 0 else in_channels
            conv_blocks.append(ConvolutionalBlock(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, stride=
                1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation=
                'LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = paddle.nn.Sequential(*conv_blocks)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.adaptive_pool = paddle.nn.AdaptiveAvgPool2D(output_size=(6, 6))
        self.fc1 = paddle.nn.Linear(in_features=out_channels * 6 * 6,
            out_features=fc_size)
        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = paddle.nn.Linear(in_features=1024, out_features=1)

    def forward(self, imgs):
        """
        前向传播.

        参数 imgs: 用于作判别的原始高清图或超分重建图，张量表示，大小为(N, 3, w * scaling factor, h * scaling factor)
        返回: 一个评分值， 用于判断一副图像是否是高清图, 张量表示，大小为 (N)
        """
        batch_size = imgs.shape[0]
        output = self.conv_blocks(imgs)
        output = self.dropout(output)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)
        return logit


class DnCNN(paddle.nn.Layer):

    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(paddle.nn.Conv2D(in_channels=channels, out_channels=
            features, kernel_size=kernel_size, padding=padding, bias_attr=
            False))
        layers.append(paddle.nn.ReLU())
        for _ in range(num_of_layers - 2):
            layers.append(paddle.nn.Conv2D(in_channels=features,
                out_channels=features, kernel_size=kernel_size, padding=
                padding, bias_attr=False))
            layers.append(paddle.nn.BatchNorm2D(num_features=features))
            layers.append(paddle.nn.ReLU())
        layers.append(paddle.nn.Conv2D(in_channels=features, out_channels=
            channels, kernel_size=kernel_size, padding=padding, bias_attr=
            False))
        self.dncnn = paddle.nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class TruncatedVGG19(paddle.nn.Layer):
    """
        truncated VGG19网络，用于计算VGG特征空间的MSE损失
        """

    def __init__(self, i, j):
        """
            :参数 i: 第 i 个池化层
            :参数 j: 第 j 个卷积层
            """
        super(TruncatedVGG19, self).__init__()
        vgg19 = paddle.vision.models.vgg19(pretrained=True)
        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1
            if isinstance(layer, paddle.nn.Conv2D):
                conv_counter += 1
            if isinstance(layer, paddle.nn.MaxPool2D):
                maxpool_counter += 1
                conv_counter = 0
            if maxpool_counter == i - 1 and conv_counter == j:
                break
        assert maxpool_counter == i - 1 and conv_counter == j, '当前 i=%d 、 j=%d 不满足 VGG19 模型结构' % (
            i, j)
        self.truncated_vgg19 = paddle.nn.Sequential(*list(vgg19.features.
            children())[:truncate_at + 1])

    def forward(self, input):
        """
            前向传播
            参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
            返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
            """
        output = self.truncated_vgg19(input)
        return output
