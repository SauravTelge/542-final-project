import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torch.utils.data import DataLoader
from kitti_dataset import KITTI
from loss import get_flow_loss, realEPE
from tqdm import trange, tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from Sintel_dataset import make_dataset, ListDataset, make_dataset_eval


parser = argparse.ArgumentParser(
        description='''Train U-Net for Optical Flow.''')
    
parser.add_argument('--data', nargs='+', type=str, default='data',
                    help='''path to the training data folder''')
parser.add_argument('--save_path', default='checkpoints_OF_mse', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--dataset', default='kitti', type=str, 
                    help='dataset')
parser.add_argument('--log_path', default='tensor_OF_mse', type=str, 
                    help='tensorboard log path')
parser.add_argument('--epochs', default=1000, type=int, 
                    help='number of epochs')
args = parser.parse_args()

log_path = args.log_path
if not os.path.exists(log_path):
    os.makedirs(log_path)

writer = SummaryWriter(log_path)

if args.dataset == 'kitti':
    training_data = KITTI(args.data)
else:
    feature_list, flow_list = make_dataset_eval(args.data, dataset_type="clean")
    training_data = ListDataset(args.data, feature_list, flow_list)

checkpoint_path = args.save_path

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
num_epochs = args.epochs
model = Model().cuda()
optimizer = optim.Adam(list(model.parameters()), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)

for epoch in range(num_epochs): 
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    if epoch==0:
        prev_loss = 0
    else:
        prev_loss = running_loss
    running_loss = 0.0
    for j, data in enumerate(pbar):
        if args.dataset == 'kitti':
            inputs, targets = data['features'], data['flow_2d'].cuda()

            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = get_flow_loss(outputs, targets, type='mse')
        else:
            inputs, targets = data['features'], data['flow'].cuda()

        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = realEPE(outputs,targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()/len(pbar)
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        pbar.set_postfix(Loss= loss.item(), Prev_epoch_loss=prev_loss)
            
    writer.add_scalar('training loss', running_loss, epoch)
    if (((epoch + 1) % 5)==0):
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, os.path.join(checkpoint_path, f'ckpt_{epoch + 1}.pt'))

print('Finished Training')