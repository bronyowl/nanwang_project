# 主要功能：定义和训练 SRResNet 模型，用于图像超分辨率。

import paddle
from models import SRResNet
from datasets import SRDataset
from utils import *
from visualdl import LogWriter

data_folder = 'C:/Users/cbbis/Desktop/Data_fusion/data'
crop_size = 100
scaling_factor = 10
scaling_factors = 8
target_size = 100, 100
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
checkpoint = None
batch_size = 64
start_epoch = 1
epochs = 180
workers = 4
lr = 0.0001
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')
ngpu = 1
#False = True
writer = LogWriter()


def main():
    """
    训练.
    """
    global checkpoint, start_epoch
    model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size
        =small_kernel_size, n_channels=n_channels, n_blocks=n_blocks,
        scaling_factors=scaling_factors)
    optimizer = paddle.optimizer.Adam(parameters=filter(lambda p: not p.
        stop_gradient, model.parameters()), learning_rate=lr, weight_decay=0.0)
    model = model.to(device)
    criterion = paddle.nn.MSELoss().to(device)
    if checkpoint is not None:
        checkpoint = paddle.load(path=checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.set_state_dict(state_dict=checkpoint['model'])
        optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
    if paddle.device.cuda.device_count() >= 1 and ngpu > 1:
        model = paddle.DataParallel(layers=model)
    train_dataset = SRDataset(data_folder, split='train', crop_size=
        crop_size, scaling_factor=scaling_factor, lr_img_type=
        'imagenet-norm', hr_img_type='[-1, 1]')
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=
        batch_size, shuffle=True, num_workers=workers)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_epoch = AverageMeter()
        n_iter = len(train_loader)
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.update(loss.item(), lr_imgs.shape[0])
            if i == n_iter - 2:
                writer.add_image('SRResNet/epoch_' + str(epoch) + '_1',
                    make_grid(lr_imgs[:4, :3, :, :].cpu().numpy(),
                    nrow=4, normalize=True), epoch)
                writer.add_image('SRResNet/epoch_' + str(epoch) + '_2',
                    make_grid(sr_imgs[:4, :3, :, :].cpu().numpy(),
                    nrow=4, normalize=True), epoch)
                writer.add_image('SRResNet/epoch_' + str(epoch) + '_3',
                    make_grid(hr_imgs[:4, :3, :, :].cpu().numpy(),
                    nrow=4, normalize=True), epoch)
            print('第 ' + str(i) + ' 个batch训练结束')
        del lr_imgs, hr_imgs, sr_imgs
        writer.add_scalar('SRResNet/MSE_Loss', loss_epoch.val, epoch)
        paddle.save(obj={'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, path=
            'C:/Users\\cbbis\\Desktop\\Data_fusion/x10/results_shuff2/checkpoint_srresnet.pth'
            )
    writer.close()


if __name__ == '__main__':
    main()
