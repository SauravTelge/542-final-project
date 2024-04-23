import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torch.utils.data import DataLoader
from datasets.kitti import KITTI_Train
from loss import get_depth_loss
from tqdm import trange, tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(
        description='''Train U-Net for Optical Flow.''')
    
parser.add_argument('--data', nargs='+', type=str, default='../data/images',
                    help='''path to the training data images folder''')
parser.add_argument('--anno_data', nargs='+', type=str, default='../data/annotations',
                    help='''path to the training data images folder''')
parser.add_argument('--save_path', default='../checkpoints/mse', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--log_path', default='../tensor_logs/mse', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--epochs', default=1000, type=int, 
                    help='number of epochs')
args = parser.parse_args()



training_data = KITTI_Train(args.data, args.anno_data)
# checkpoint_path = args.save_path

# # training_data = KITTI(r'/nfs/turbo/jjparkcv-turbo-large/instant3d/detection/dift/data')
# # checkpoint_path = r'/nfs/turbo/jjparkcv-turbo-large/instant3d/detection/dift/checkpoints'

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

writer = SummaryWriter(args.log_path)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
# sample = next(iter(train_dataloader))
# print(sample.keys())
# print(sample['depth'].shape)

num_epochs = args.epochs
model = Model().cuda()
optimizer = optim.Adam(list(model.parameters()), lr=1e-2, betas=(0.9, 0.98), eps=1e-9)
scale = 90.0


for epoch in range(num_epochs): 
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    if epoch==0:
        prev_loss = 0
    else:
        prev_loss = running_loss
    running_loss = 0.0
    for j, data in enumerate(pbar):

        # if j<5:
        
            inputs, targets = data['features'], data['depth'].cuda()

            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
            # print(targets[:, :2, :, :].max().item(), targets[:, :2, :, :].min().item(), 'flow range')
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = get_depth_loss(outputs, targets, type='mse')
            loss.backward()
            optimizer.step()
            running_loss += loss.item()/len(pbar)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            pbar.set_postfix(Loss= loss.item(), Prev_epoch_loss=prev_loss)
            # pbar.set_postfix(loss_step=loss.item(), prev_epoch_loss=prev_loss)
    writer.add_scalar('training loss', running_loss, epoch)
    if (((epoch + 1) % 5)==0):
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, os.path.join(args.save_path, f'ckpt_{epoch + 1}.pt'))

print('Finished Training')