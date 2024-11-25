# 主要功能：定义和训练生成器和判别器，实现 SRGAN 的训练过程。

import paddle
from models import Generator, Discriminator
from datasets import SRDataset
from utils import *
from visualdl import LogWriter

data_folder = './data'
crop_size = 100
scaling_factors = 8
scaling_factor = 10
target_size = 100, 100
large_kernel_size_g = 9
small_kernel_size_g = 3
n_channels_g = 64
n_blocks_g = 5
srresnet_checkpoint = './checkpoints/srresnet/srresnet.pdparams'
kernel_size_d = 3
n_channels_d = 64
n_blocks_d = 8
fc_size_d = 1024
batch_size = 128
start_epoch = 1
epochs = 50
checkpoint = None
workers = 0
beta = 0.001
lr = 0.001
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')
ngpu = 1
#False = True
writer = LogWriter()


def main():
    """
    训练.
    """
    global checkpoint, start_epoch, writer
    generator = Generator(large_kernel_size=large_kernel_size_g,
        small_kernel_size=small_kernel_size_g, n_channels=n_channels_g,
        n_blocks=n_blocks_g, scaling_factors=scaling_factors, target_size=target_size)
    discriminator = Discriminator(kernel_size=kernel_size_d, n_channels=
        n_channels_d, n_blocks=n_blocks_d, fc_size=fc_size_d)
    optimizer_g = paddle.optimizer.Adam(parameters=filter(lambda p: not p.
        stop_gradient, generator.parameters()), learning_rate=lr,
        weight_decay=1e-05)
    optimizer_d = paddle.optimizer.Adam(parameters=filter(lambda p: not p.
        stop_gradient, discriminator.parameters()), learning_rate=lr,
        weight_decay=1e-05)
    content_loss_criterion = paddle.nn.MSELoss()
    adversarial_loss_criterion = paddle.nn.BCEWithLogitsLoss()
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)
    srresnetcheckpoint = paddle.load(path=srresnet_checkpoint)
    generator.set_net_state_dict(srresnetcheckpoint['model'])
    if checkpoint is not None:
        checkpoint = paddle.load(path=checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator.set_state_dict(state_dict=checkpoint['generator'])
        discriminator.set_state_dict(state_dict=checkpoint['discriminator'])
        optimizer_g.set_state_dict(state_dict=checkpoint['optimizer_g'])
        optimizer_d.set_state_dict(state_dict=checkpoint['optimizer_d'])
    if paddle.device.cuda.device_count() >= 1 and ngpu > 1:
        generator = paddle.DataParallel(layers=generator)
        discriminator = paddle.DataParallel(layers=discriminator)
    train_dataset = SRDataset(data_folder, split='train', crop_size=
        crop_size, scaling_factor=scaling_factor, lr_img_type=
        'imagenet-norm', hr_img_type='imagenet-norm')
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=
        batch_size, shuffle=True, num_workers=workers, places=device)
    for epoch in range(start_epoch, epochs + 1):
        if epoch == int(epochs / 2):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)
            print(f'Learning rate adjusted at epoch {epoch}')
        generator.train()
        discriminator.train()
        losses_c = AverageMeter()
        losses_a = AverageMeter()
        losses_d = AverageMeter()
        n_iter = len(train_loader)
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target=
                'imagenet-norm')
            content_loss = content_loss_criterion(sr_imgs, hr_imgs)
            sr_discriminated = discriminator(sr_imgs)
            adversarial_loss = adversarial_loss_criterion(sr_discriminated,
                paddle.ones_like(x=sr_discriminated))
            perceptual_loss = content_loss + beta * adversarial_loss
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            optimizer_g.clear_grad()
            perceptual_loss.backward()
            optimizer_g.step()
            losses_c.update(content_loss.item(), lr_imgs.shape[0])
            losses_a.update(adversarial_loss.item(), lr_imgs.shape[0])
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())
            adversarial_loss = adversarial_loss_criterion(sr_discriminated,
                paddle.zeros_like(x=sr_discriminated)
                ) + adversarial_loss_criterion(hr_discriminated, paddle.
                ones_like(x=hr_discriminated))
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            optimizer_d.clear_grad()
            adversarial_loss.backward()
            optimizer_d.step()
            losses_d.update(adversarial_loss.item(), hr_imgs.shape[0])
            if i == n_iter - 2:
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_1',
                    make_grid(lr_imgs[:4, :3, :, :].cpu().numpy(),
                    nrow=4, normalize=True), epoch)
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_2',
                    make_grid(sr_imgs[:4, :3, :, :].cpu().numpy(),
                    nrow=4, normalize=True), epoch)
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_3',
                    make_grid(hr_imgs[:4, :3, :, :].cpu().numpy(),
                    nrow=4, normalize=True), epoch)
            print('第 ' + str(i) + ' 个batch结束')
        del lr_imgs, hr_imgs, sr_imgs, hr_discriminated, sr_discriminated
        writer.add_scalar('SRGAN/Loss_c', losses_c.val, epoch)
        writer.add_scalar('SRGAN/Loss_a', losses_a.val, epoch)
        writer.add_scalar('SRGAN/Loss_d', losses_d.val, epoch)
        paddle.save(obj={'epoch': epoch, 'generator': generator.state_dict(
            ), 'discriminator': discriminator.state_dict(), 'optimizer_g':
            optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict
            ()}, path=
            './checkpoints/srgan/checkpoint_srgan.pdparams'
            )
        writer.close()


if __name__ == '__main__':
    main()


