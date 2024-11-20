# 主要功能：定义模型架构，包含了 ConvolutionalBlock、SubPixelConvolutionalBlock、ResidualBlock 和 SRResNet 等模型组件和完整模型结构。

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
        =64, n_blocks=5, scaling_factors=8, target_size=(100, 100)):
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
        self.residual_blocks = paddle.nn.Sequential(*[ResidualBlock(
            kernel_size=small_kernel_size, n_channels=n_channels) for i in
            range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels,
            out_channels=n_channels, kernel_size=small_kernel_size,
            batch_norm=True, activation=None)
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
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        output = paddle.nn.functional.interpolate(x=output, size=self.
            target_size, mode='bilinear', align_corners=False)
        sr_imgs = self.conv_block3(output)
        return sr_imgs
