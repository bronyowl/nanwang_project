import paddle
from model import SRResNet
from datasets import SRDataset
from utils import *
from visualdl import LogWriter

data_folder = './data'
crop_size = 100
scaling_factor = 10
scaling_factors = 8
target_size = (100, 100)
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
checkpoint = None
batch_size = 64
start_epoch = 1
epochs = 180
workers = 0
lr = 0.0001

# Set device based on availability of GPU or fallback to CPU
device = 'gpu' if paddle.device.cuda.device_count() > 0 else 'cpu'
paddle.device.set_device(device)

ngpu = 1
writer = LogWriter(logdir='./log/srresnet')

def main():
    """
    Training.
    """
    global checkpoint, start_epoch
    
    # Initialize model and move it to the appropriate device
    model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                     n_channels=n_channels, n_blocks=n_blocks, scaling_factors=scaling_factors)
    
    optimizer = paddle.optimizer.Adam(parameters=filter(lambda p: not p.stop_gradient, model.parameters()),
                                      learning_rate=lr, weight_decay=0.0)
    
    # Move model to the selected device
    model.to(device)
    
    # Initialize loss function on the selected device
    criterion = paddle.nn.MSELoss().to(device)
    
    if checkpoint is not None:
        checkpoint_data = paddle.load(path=checkpoint)
        start_epoch = checkpoint_data['epoch'] + 1
        model.set_state_dict(state_dict=checkpoint_data['model'])
        optimizer.set_state_dict(state_dict=checkpoint_data['optimizer'])
    
    if paddle.device.cuda.device_count() > 1 and ngpu > 1:
        model = paddle.DataParallel(layers=model)
    
    train_dataset = SRDataset(data_folder, split='train', crop_size=crop_size,
                              scaling_factor=scaling_factor, lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')
    
    # Set DataLoader to use the appropriate device for loading data
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers,
                                         use_shared_memory=False)
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_epoch = AverageMeter()
        n_iter = len(train_loader)
        
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            # Move images to the selected device before processing
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            # 记录当前批次的loss
            writer.add_scalar(tag='Train/Loss', 
                            step=epoch * n_iter + i,
                            value=loss.item())
            
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
            
            print('Batch ' + str(i) + ' training completed')
        
        del lr_imgs, hr_imgs, sr_imgs
        
        # 记录每个epoch的平均loss
        writer.add_scalar(tag='Train/Epoch_Loss',
                         step=epoch,
                         value=loss_epoch.avg)
        
        paddle.save(obj={'epoch': epoch,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()},
                    path='./checkpoints/srresnet/srresnet.pdparams')
    
    writer.close()

if __name__ == '__main__':
    main()