import torch
import torch.nn as nn
from model import Model
import numpy as np
from Sintel_dataset import make_dataset, ListDataset
import argparse
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch.nn.functional as F
from imageio import imread, imwrite
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import flow_viz
import os

from loss import realEPE, mse_loss

def flow2rgb(flow_map, max_value):
    # flow_map_np = flow_map.detach().cpu().numpy()
    flow_map_np = flow_map
    _, h, w = flow_map_np.shape # 2,h,w -> 1, h, w and h,w

    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)

parser = argparse.ArgumentParser(
        description='''Visualization of Optical Flow.''')
    
parser.add_argument('--data', type=str, default='data',
                    help='''path to the data folder''')
parser.add_argument('--save_path', default='viz_results', type=str, 
                    help='viz results')
# parser.add_argument('--epochs', default=100, type=int, 
#                     help='number of epochs')
# parser.add_argument('--batch_size', default=2, type=int, 
#                     help='batch size')
parser.add_argument('--checkpoint_path', default='', 
                    help='path of saved checkpoint')
parser.add_argument(
    "--div-flow",
    default=20,
    type=float,
    help="value by which flow will be divided. overwritten if stored in pretrained file",
)
parser.add_argument(
    "--max_flow",
    default=None,
    type=float,
    help="max flow value. Flow map color is saturated above this value. If not set, will use flow map's max value",
)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



X_train, X_test, y_train, y_test = make_dataset(args.data,dataset_type="clean")
train_dataset = ListDataset(args.data, X_train, y_train)
# temp_load = train_dataset.__getitem__(0)
# print(f"Shape of loaded target: {temp_load['flow'].shape}")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataset = ListDataset(args.data, X_test, y_test)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# result_path = args.save_path

# model = Model().cuda()
model = Model().to(device=device)
# /home/rutwik/542project/dift/project_sintel_files/checkpoints/ckpt_5.pt
if args.checkpoint_path != '':
    print('Loading checkpoint')
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    # running_loss = checkpoint['loss']
    # assert start_epoch < num_epochs

i_output = 0
model.eval()
pbar = tqdm(train_dataloader)

for data in pbar:

    if i_output>4: break
    inputs, targets = data['features'], data['flow'].cuda()

    for i in range(len(inputs)):
        inputs[i] = inputs[i].cuda()
    
    # forward + backward + optimize
    outputs = model(inputs)
    # outputs = torch.transpose(outputs, 3, 2)
    # print(f"output shape: {outputs.shape}")
    # print(f"target shape: {targets.shape}")
    # loss = realEPE(outputs,targets)

    # b, _, h, w = targets.size()
    # upsampled_output = F.interpolate(
    #     outputs, (h, w), mode="bilinear", align_corners=False
    # )
    # loss_val = realEPE(upsampled_output,targets)
    loss_val = mse_loss(outputs,targets)
    print(f"Loss: {loss_val}")
    print(f"Range of output: min: {outputs.min()} max: {outputs.max()}")
    print(f"Range of target: min: {targets.min()} max: {targets.max()}")

    rgb_flow_output = flow2rgb(
                args.div_flow * outputs.squeeze(0).detach().cpu().numpy(), max_value=args.max_flow
            )
    
    to_save = (rgb_flow_output * 255).astype(np.uint8).transpose(1, 2, 0)
    # print(f"Output shape: {to_save.shape}")
    imwrite(args.save_path+f'/output_{i_output}' + ".png", to_save)
    # print(f"Saved at :"+args.save_path+f"/output_{i_output}.png")

    rgb_flow_gt = flow2rgb(
                args.div_flow * targets.squeeze(0).detach().cpu().numpy(), max_value=args.max_flow
            )
    
    to_save = (rgb_flow_gt * 255).astype(np.uint8).transpose(1, 2, 0)
    # print(f"Target shape: {to_save.shape}")
    imwrite(args.save_path+f'/target_{i_output}' + ".png", to_save)
    # print(f"Saved at :"+args.save_path+f"/target_{i_output}.png")

    i_output+=1
